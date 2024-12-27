from argparse import Namespace
import torch
import math
from typing import Union, List, Optional

from .Nvidia_Alignment_Learning_Framework import Alignment_Learning_Framework
from .Layer import Conv_Init, Embedding_Initialize_, FFT_Block, Norm_Type
from .GRL import GRL
from .CFM import CFM

from hificodec.vqvae import VQVAE

class RectifiedFlowTTS(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        hificodec: VQVAE
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)
        
        self.alignment_learning_framework = Alignment_Learning_Framework(
            feature_size= self.hp.Sound.Num_Mel,
            encoding_size= self.hp.Encoder.Size
            )
        
        self.prompt_encoder = Prompt_Encoder(self.hp)

        self.duration_predictor = Duration_Predictor(self.hp)
        self.f0_predictor = F0_Predictor(self.hp)
        self.speaker_eliminator = Eliminator(self.hp, self.hp.Speakers)

        self.frame_prior_network = Frame_Prior_Network(self.hp)

        self.cfm = CFM(self.hp)

        self.hificodec = hificodec
        self.hificodec.eval()

    def forward(
        self,
        tokens: torch.LongTensor,
        token_lengths: torch.LongTensor,
        reference_latent_codes: torch.LongTensor,
        reference_latent_code_lengths: torch.LongTensor,
        languages: torch.LongTensor,
        latent_codes: torch.LongTensor,
        latent_code_lengths: torch.LongTensor,
        f0s: torch.FloatTensor,
        mels: torch.FloatTensor,
        attention_priors: torch.FloatTensor
        ):
        with torch.no_grad():
            latents = self.hificodec.quantizer.embed(latent_codes.mT).detach()
            reference_latents = self.hificodec.quantizer.embed(reference_latent_codes.mT).detach()

        prompts = self.prompt_encoder(
            latents= reference_latents,
            lengths= reference_latent_code_lengths,
            )    # [Batch, Enc_d, Prompt_t]
        encodings = self.encoder(
            tokens= tokens,
            languages= languages,
            lengths= token_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )    # [Batch, Enc_d, Enc_t]        

        durations, attention_softs, attention_hards, attention_logprobs = self.alignment_learning_framework(
            token_embeddings= self.encoder.token_embedding(tokens).permute(0, 2, 1),
            encoding_lengths= token_lengths,
            features= mels,
            feature_lengths= latent_code_lengths,
            attention_priors= attention_priors
            )
        prediction_durations = self.duration_predictor(
            encodings= encodings,
            lengths= token_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )   # [Batch, Enc_t]
        prediction_speakers = self.speaker_eliminator(encodings)
        
        alignments = self.Length_Regulate(durations)
        encodings = encodings @ alignments  # [Batch, Enc_d, Dec_t]
        
        encodings = self.frame_prior_network.forward(
            encodings= encodings,
            lengths= latent_code_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )   # [Batch, Enc_d, Dec_t]
        prediction_f0s = self.f0_predictor(
            encodings= encodings,
            lengths= latent_code_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )   # [Batch, Dec_t]

        flows, prediction_flows, _, _, prediction_tokens = self.cfm(
            encodings= encodings,
            f0s= f0s,
            latents= latents,
            lengths= latent_code_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )

        return \
            flows, prediction_flows, \
            durations, prediction_durations, prediction_f0s, \
            attention_softs, attention_hards, attention_logprobs, \
            prediction_speakers, prediction_tokens, alignments
    
    def Inference(
        self,
        tokens: torch.LongTensor,
        token_lengths: torch.LongTensor,
        reference_latent_codes: torch.LongTensor,
        reference_latent_code_lengths: torch.LongTensor,
        languages: torch.LongTensor,
        cfm_steps: int= 16
        ):
        with torch.no_grad():
            reference_latents = self.hificodec.quantizer.embed(reference_latent_codes.mT).detach()

        prompts = self.prompt_encoder(
            latents= reference_latents,
            lengths= reference_latent_code_lengths,
            )    # [Batch, Enc_d, Prompt_t]
        encodings = self.encoder(
            tokens= tokens,
            languages= languages,
            lengths= token_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )    # [Batch, Enc_d, Enc_t]

        durations = self.duration_predictor(
            encodings= encodings,
            lengths= token_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            ).ceil().long()   # [Batch, Enc_t]
        alignments = self.Length_Regulate(durations)

        encodings = encodings @ alignments  # [Batch, Enc_d, Dec_t]

        latent_code_lengths = torch.stack([
            alignment[:token_length, :].sum().long()
            for token_length, alignment in zip(token_lengths, alignments)
            ])

        encodings = self.frame_prior_network(
            encodings= encodings,
            lengths= latent_code_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            ) # [Batch, Enc_d, Dec_t]
        f0s = self.f0_predictor(
            encodings= encodings,
            lengths= latent_code_lengths,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )   # [Batch, Dec_t]

        latents = self.cfm.Inference(
            encodings= encodings,
            f0s= f0s,
            lengths= latent_code_lengths,
            steps= cfm_steps,
            prompts= prompts,
            prompt_lengths= reference_latent_code_lengths
            )

        # Performing VQ to correct the incomplete predictions.
        *_, latent_codes = self.hificodec.quantizer(latents)
        latent_codes = [code.reshape(tokens.size(0), -1) for code in latent_codes]
        latent_codes = torch.stack(latent_codes, 2)
        audios = self.hificodec(latent_codes)[:, 0, :]

        return audios, f0s, alignments

    def Length_Regulate(
        self,
        durations: torch.IntTensor
        ) -> torch.FloatTensor:
        repeats = (durations.float() + 0.5).long()
        decoding_lengths = repeats.sum(dim=1)

        max_decoding_length = decoding_lengths.max()
        reps_cumsum = torch.cumsum(torch.nn.functional.pad(repeats, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]

        range_ = torch.arange(max_decoding_length)[None, :, None].to(durations.device)
        alignments = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)

        return alignments.permute(0, 2, 1).float()
        
    def train(self, mode: bool= True):
        super().train(mode= mode)
        self.hificodec.eval()


class Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size,
            )
        Embedding_Initialize_(self.token_embedding)

        self.language_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Languages,
            embedding_dim= self.hp.Encoder.Size,
            )
        Embedding_Initialize_(self.language_embedding)
        
        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Transformer.Head,
                ffn_kernel_size= self.hp.Encoder.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Encoder.Transformer.FFN.Dropout_Rate,
                norm_type= Norm_Type.Conditional_LayerNorm,
                condition_channels= self.hp.Encoder.Size,
                )
            for _ in range(self.hp.Encoder.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        tokens: torch.FloatTensor,
        languages: torch.LongTensor,
        lengths: torch.LongTensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor
        ) -> torch.FloatTensor:
        '''
        tokens: [Batch, Enc_t]
        languages: [Batch] or [Batch, Enc_t]
        lengths: [Batch], token length
        '''
        encodings = self.token_embedding(tokens)    # [Batch, Enc_t, Enc_d]

        languages = self.language_embedding(languages)  # [Batch, Enc_d] or [Batch, Enc_t, Enc_d]
        if languages.ndim == 2:
            languages = languages[:, None, :]  # [Batch, 1, Enc_d]
        encodings = encodings + languages

        encodings = encodings.mT # [Batch, Enc_d, Enc_t]        
        
        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths,
                conditions= prompts,
                condition_lengths= prompt_lengths
                )
            
        return encodings

class Prompt_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters
 
        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Audio_Codec_Size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            ), w_init_gain= 'gelu')
        self.gelu = torch.nn.GELU(approximate= 'tanh')
        
        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Prompter.Transformer.Head,
                ffn_kernel_size= self.hp.Prompter.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Prompter.Transformer.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.Prompter.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]
    
    def forward(
        self,
        latents: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Enc_t]        
        lengths: [Batch], token length
        '''
        latents = self.prenet(latents)
        latents = self.gelu(latents)
        
        for block in self.blocks:
            latents = block(latents, lengths)

        return latents

class Duration_Predictor(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters
 
        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Duration_Predictor.Transformer.Head,
                ffn_kernel_size= self.hp.Duration_Predictor.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Duration_Predictor.Transformer.FFN.Dropout_Rate,
                norm_type= Norm_Type.Conditional_LayerNorm,
                condition_channels= self.hp.Encoder.Size,
                )
            for _ in range(self.hp.Duration_Predictor.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= 1,
            kernel_size= 1
            ), w_init_gain= 'softplus')
    
    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor
        ) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Enc_t]        
        lengths: [Batch], token length
        '''
        encodings = encodings.detach()

        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths,
                conditions= prompts,
                condition_lengths= prompt_lengths
                )

        durations = self.projection(encodings)[:, 0, :] # [Batch, Enc_t]

        return torch.nn.functional.softplus(durations)
    
class F0_Predictor(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.F0_Predictor.Transformer.Head,
                ffn_kernel_size= self.hp.F0_Predictor.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.F0_Predictor.Transformer.FFN.Dropout_Rate,
                norm_type= Norm_Type.Conditional_LayerNorm,
                condition_channels= self.hp.Encoder.Size,
                )
            for _ in range(self.hp.F0_Predictor.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]
        
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= 1,
            kernel_size= 1
            ), w_init_gain= 'softplus')
    
    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor
        ) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Dec_t]        
        lengths: [Batch], latent length
        '''
        encodings = encodings.detach()
        
        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths,
                conditions= prompts,
                condition_lengths= prompt_lengths
                )

        f0s = self.projection(encodings)[:, 0, :] # [Batch, Dec_t]

        return torch.nn.functional.softplus(f0s)

class Eliminator(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace,
        num_classes: int
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.grl = GRL()

        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Eliminator.Size,
            kernel_size= self.hp.Eliminator.Kernel_Size
            ), w_init_gain= 'relu')
        self.norm_0 = torch.nn.LayerNorm(self.hp.Eliminator.Size)

        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Eliminator.Size,
            out_channels= self.hp.Eliminator.Size,
            kernel_size= self.hp.Eliminator.Kernel_Size
            ), w_init_gain= 'relu')
        self.norm_1 = torch.nn.LayerNorm(self.hp.Eliminator.Size)

        self.conv_2 = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Eliminator.Size,
            out_channels= self.hp.Eliminator.Size,
            kernel_size= self.hp.Eliminator.Kernel_Size
            ), w_init_gain= 'relu')
        self.norm_2 = torch.nn.LayerNorm(self.hp.Eliminator.Size)

        self.projection = Conv_Init(torch.nn.Linear(
            in_features= self.hp.Eliminator.Size,
            out_features= num_classes
            ), w_init_gain= 'linear')

        self.gelu = torch.nn.GELU(approximate= 'tanh')
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grl(x)
        x = self.conv_0(x)
        x = self.norm_0(x.mT).mT
        x = self.gelu(x)
        x = self.conv_1(x)
        x = self.norm_1(x.mT).mT
        x = self.gelu(x)
        x = self.conv_2(x)
        x = self.norm_2(x.mT).mT
        x = self.gelu(x).mean(dim= 2)
        x = self.projection(x)

        return x

class Frame_Prior_Network(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Frame_Prior_Network.Transformer.Head,
                ffn_kernel_size= self.hp.Frame_Prior_Network.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Frame_Prior_Network.Transformer.FFN.Dropout_Rate,
                norm_type= Norm_Type.Conditional_LayerNorm,
                condition_channels= self.hp.Encoder.Size,
                )
            for _ in range(self.hp.Frame_Prior_Network.Transformer.Stack)
            ])

    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor
        ) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Dec_t],
        lengths: [Batch], latent length
        '''
        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths,
                conditions= prompts,
                condition_lengths= prompt_lengths
                )
        
        return encodings


def Mask_Generate(lengths: torch.Tensor, max_length: Union[torch.Tensor, int, None]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]