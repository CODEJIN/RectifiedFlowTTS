from argparse import Namespace
import torch
import math
from typing import Union, List, Optional

from .Nvidia_Alignment_Learning_Framework import Alignment_Learning_Framework
from .Layer import Conv_Init, Embedding_Initialize_, FFT_Block, Norm_Type
from .GRL import GRL
from .CFM import CFM

class RectifiedFlowTTS(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        mel_min: float,
        mel_max: float
        ):
        super().__init__()
        self.hp = hyper_parameters
        self.mel_min = mel_min
        self.mel_max = mel_max
        
        self.prompt_encoder = Prompt_Encoder(self.hp)
        self.encoder = Encoder(self.hp)
        
        self.alignment_learning_framework = Alignment_Learning_Framework(
            feature_size= self.hp.Sound.N_Mel,
            encoding_size= self.hp.Encoder.Size
            )

        self.duration_predictor = Duration_Predictor(self.hp)
        self.f0_predictor = F0_Predictor(self.hp)

        self.frame_prior_network = Frame_Prior_Network(self.hp)
        self.linear_projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Sound.N_Mel,
            kernel_size= 1
            ), w_init_gain= 'linear')

        self.cfm = CFM(self.hp)

    def forward(
        self,
        tokens: torch.LongTensor,
        token_lengths: torch.LongTensor,
        reference_mels: torch.LongTensor,
        reference_mel_lengths: torch.LongTensor,
        languages: torch.LongTensor,
        mels: torch.LongTensor,
        mel_lengths: torch.LongTensor,
        f0s: torch.FloatTensor,
        attention_priors: torch.FloatTensor
        ):
        target_mels = (mels - self.mel_min) / (self.mel_max - self.mel_min) * 2.0 - 1.0
        reference_mels = (reference_mels - self.mel_min) / (self.mel_max - self.mel_min) * 2.0 - 1.0

        prompts = self.prompt_encoder(
            mels= reference_mels,
            lengths= reference_mel_lengths,
            )    # [Batch, Enc_d, Prompt_t]
        encodings = self.encoder(
            tokens= tokens,
            languages= languages,
            lengths= token_lengths,
            )    # [Batch, Enc_d, Enc_t]

        durations, attention_softs, attention_hards, attention_logprobs = self.alignment_learning_framework(
            token_embeddings= torch.cat([
                self.encoder.token_embedding(tokens), 
                self.encoder.language_embedding(languages)
                ], dim= 2).mT,
            encoding_lengths= token_lengths,
            features= mels,
            feature_lengths= mel_lengths,
            attention_priors= attention_priors
            )
        prediction_durations = self.duration_predictor(
            encodings= encodings,
            lengths= token_lengths,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            )   # [Batch, Enc_t]        
        alignments = self.Length_Regulate(durations)
        encodings = encodings @ alignments  # [Batch, Enc_d, Dec_t]
        
        encodings = self.frame_prior_network(
            encodings= encodings,
            lengths= mel_lengths,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            )   # [Batch, Enc_d, Dec_t]
        prediction_f0s = self.f0_predictor(
            encodings= encodings,
            lengths= mel_lengths,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            )   # [Batch, Dec_t]

        linear_prediction_mels = self.linear_projection(encodings)

        flows, prediction_flows = self.cfm(
            encodings= linear_prediction_mels,
            f0s= f0s,
            mels= target_mels,
            lengths= mel_lengths,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            )

        return \
            flows, prediction_flows, \
            target_mels, linear_prediction_mels, \
            durations, prediction_durations, prediction_f0s, \
            attention_softs, attention_hards, attention_logprobs, \
            alignments
    
    def Inference(
        self,
        tokens: torch.LongTensor,
        token_lengths: torch.LongTensor,
        reference_mels: torch.LongTensor,
        reference_mel_lengths: torch.LongTensor,
        languages: torch.LongTensor,
        cfm_steps: int= 16
        ):
        reference_mels = (reference_mels - self.mel_min) / (self.mel_max - self.mel_min) * 2.0 - 1.0

        prompts = self.prompt_encoder(
            mels= reference_mels,
            lengths= reference_mel_lengths,
            )    # [Batch, Enc_d, Prompt_t]
        encodings = self.encoder(
            tokens= tokens,
            languages= languages,
            lengths= token_lengths,
            )    # [Batch, Enc_d, Enc_t]

        durations = self.duration_predictor(
            encodings= encodings,
            lengths= token_lengths,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            ).ceil().long()   # [Batch, Enc_t]
        alignments = self.Length_Regulate(durations)
        encodings = encodings @ alignments  # [Batch, Enc_d, Dec_t]

        mel_lengths = torch.stack([
            alignment[:token_length, :].sum().long()
            for token_length, alignment in zip(token_lengths, alignments)
            ])

        encodings = self.frame_prior_network(
            encodings= encodings,
            lengths= mel_lengths,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            ) # [Batch, Enc_d, Dec_t]
        f0s = self.f0_predictor(
            encodings= encodings,
            lengths= mel_lengths,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            )   # [Batch, Dec_t]

        linear_prediction_mels = self.linear_projection(encodings)

        mels = self.cfm.Inference(
            encodings= linear_prediction_mels,
            f0s= f0s,
            lengths= mel_lengths,
            steps= cfm_steps,
            prompts= prompts,
            prompt_lengths= reference_mel_lengths
            )
        
        mels = (mels + 1.0) / 2.0 * (self.mel_max - self.mel_min) + self.mel_min
        linear_prediction_mels = (linear_prediction_mels + 1.0) / 2.0 * (self.mel_max - self.mel_min) + self.mel_min

        return mels, f0s, alignments, linear_prediction_mels

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


class Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size // 2,
            )
        Embedding_Initialize_(self.token_embedding)

        self.language_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Languages,
            embedding_dim= self.hp.Encoder.Size // 2,
            )
        Embedding_Initialize_(self.language_embedding)
        
        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Transformer.Head,
                ffn_kernel_size= self.hp.Encoder.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Encoder.Transformer.FFN.Dropout_Rate,
                norm_type= Norm_Type.LayerNorm,
                )
            for index in range(self.hp.Encoder.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        tokens: torch.FloatTensor,
        languages: torch.LongTensor,
        lengths: torch.LongTensor,
        ) -> torch.FloatTensor:
        '''
        tokens: [Batch, Enc_t]
        languages: [Batch] or [Batch, Enc_t]
        lengths: [Batch], token length
        '''
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(tokens[0]).sum())    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]

        tokens = self.token_embedding(tokens)    # [Batch, Enc_t, Enc_d]
        languages = self.language_embedding(languages)  # [Batch, Enc_d] or [Batch, Enc_t, Enc_d]
        encodings = torch.cat([tokens, languages], dim= 2).mT * float_masks # [Batch, Enc_d, Token_t]
        
        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths,
                )
            
        return encodings * float_masks

class Prompt_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters
 
        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Sound.N_Mel,
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
                norm_type= Norm_Type.LayerNorm,
                )
            for index in range(self.hp.Prompter.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]
    
    def forward(
        self,
        mels: torch.Tensor,
        lengths: torch.Tensor,
        ) -> torch.Tensor:
        '''
        encodings: [Batch, Enc_d, Enc_t]        
        lengths: [Batch], token length
        '''
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(mels[0, 0]).sum())    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]

        mels = self.prenet(mels)
        mels = self.gelu(mels) * float_masks
        
        for block in self.blocks:
            mels = block(mels, lengths)

        return mels * float_masks

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
                norm_type= Norm_Type.LayerNorm,
                cross_attention_condition_channels= self.hp.Encoder.Size if index == 0 else None,
                )
            for index in range(self.hp.Duration_Predictor.Transformer.Stack)
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

        for index, block in enumerate(self.blocks):
            encodings = block(
                x= encodings,
                lengths= lengths,
                cross_attention_conditions= prompts if index == 0 else None,
                cross_attention_condition_lengths= prompt_lengths if index == 0 else None,
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
                norm_type= Norm_Type.LayerNorm,
                cross_attention_condition_channels= self.hp.Encoder.Size if index == 0 else None,
                )
            for index in range(self.hp.F0_Predictor.Transformer.Stack)
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
        lengths: [Batch], mel length
        '''
        encodings = encodings.detach()
            
        for index, block in enumerate(self.blocks):
            encodings = block(
                x= encodings,
                lengths= lengths,
                cross_attention_conditions= prompts if index == 0 else None,
                cross_attention_condition_lengths= prompt_lengths if index == 0 else None,
                )

        f0s = self.projection(encodings)[:, 0, :] # [Batch, Dec_t]

        return torch.nn.functional.softplus(f0s)

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
                norm_type= Norm_Type.LayerNorm,
                cross_attention_condition_channels= self.hp.Encoder.Size if index == 0 else None,
                )
            for index in range(self.hp.Frame_Prior_Network.Transformer.Stack)
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
        lengths: [Batch], mel length
        '''
        for index, block in enumerate(self.blocks):
            encodings = block(
                x= encodings,
                lengths= lengths,
                cross_attention_conditions= prompts if index == 0 else None,
                cross_attention_condition_lengths= prompt_lengths if index == 0 else None,
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