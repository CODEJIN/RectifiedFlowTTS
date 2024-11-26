import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from torchdiffeq import odeint

from .Layer import Conv_Init, Lambda, FFT_Block, Prompt_Block

class Diffusion(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.network = Network(
            hyper_parameters= self.hp
            )
        self.cosmap_calc = lambda x: 1.0 - (1 / (torch.tan(torch.pi / 2.0 * x) + 1))

    def forward(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
        latents: torch.FloatTensor,
        lengths: torch.IntTensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor
        ):
        '''
        encodings: [Batch, Enc_d, Dec_t]
        latents: [Batch, Latent_d, Latent_t]
        '''
        diffusion_times = torch.rand(encodings.size(0), device= encodings.device)   # [Batch]
        cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None, None]

        noises = torch.randn_like(latents)  # [Batch, Latent_d, Latent_t]

        noised_latents = cosmap_schedule_times * latents + (1.0 - cosmap_schedule_times) * noises
        flows = latents - noises

        # predict flow
        network_output, prediction_tokens = self.network(
            noised_latents= noised_latents,
            encodings= encodings,
            f0s= f0s,
            lengths= lengths,
            prompts= prompts,
            prompt_lengths= prompt_lengths,
            diffusion_steps= cosmap_schedule_times[:, 0, 0], # [Batch]
            use_token_predictor= True
            )
        
        if self.hp.Diffusion.Network_Prediction == 'Flow':
            prediction_flows = network_output
            prediction_noises = noised_latents - prediction_flows * cosmap_schedule_times.clamp(1e-7)   # is it working?
        elif self.hp.Diffusion.Network_Prediction == 'Noise':
            prediction_noises = network_output
            prediction_flows = (noised_latents - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
        else:
            NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.Diffusion.Network_Prediction}')

        # prediction_latents = noised_latents + prediction_flows * (1.0 - cosmap_schedule_times)    # not using

        return flows, prediction_flows, noises, prediction_noises, prediction_tokens
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
        lengths: torch.IntTensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor,
        steps: int
        ):
        noises = torch.randn(
            encodings.size(0), self.hp.Audio_Codec_Size, encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, Latent_d, Latent_t]
        
        def ode_func(diffusion_times: torch.Tensor, noised_latents: torch.Tensor):
            diffusion_times = diffusion_times[None].expand(noised_latents.size(0))
            cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None, None]

            # predict flow
            network_output, _ = self.network(
                noised_latents= noised_latents,
                encodings= encodings,
                f0s= f0s,
                lengths= lengths,
                prompts= prompts,
                prompt_lengths= prompt_lengths,
                diffusion_steps= cosmap_schedule_times[:, 0, 0] # [Batch]
                )
            
            if self.hp.Diffusion.Network_Prediction == 'Flow':
                prediction_flows = network_output                
            elif self.hp.Diffusion.Network_Prediction == 'Noise':
                prediction_noises = network_output
                prediction_flows = (noised_latents - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
            else:
                NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.Diffusion.Network_Prediction}')

            return prediction_flows
        
        latents = odeint(
            func= ode_func,
            y0= noises,
            t= torch.linspace(0.0, 1.0, steps, device= encodings.device),
            atol= 1e-5,
            rtol= 1e-5,
            method= 'midpoint'
            )[-1]

        return latents

class Network(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Audio_Codec_Size,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )
        
        self.encoding_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.Diffusion.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Size * 4,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )

        self.f0_ffn = torch.nn.Sequential(
            Lambda(lambda x: x[:, :, None]),
            Conv_Init(torch.nn.Linear(
                in_features= 1,
                out_features= self.hp.Diffusion.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Size * 4,
                out_features= self.hp.Diffusion.Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x.mT)
            )
        
        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.Diffusion.Size
                ),            
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Size,
                out_features= self.hp.Diffusion.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Size * 4,
                out_features= self.hp.Diffusion.Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x[:, :, None])
            )

        self.prompt_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.Diffusion.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Size * 4,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )

        

        self.prompt_blocks: List[Prompt_Block] = torch.nn.ModuleList([
            Prompt_Block(
                channels= self.hp.Diffusion.Size,
                num_head= self.hp.Diffusion.Transformer.Head,
                residual_conv_stack= self.hp.Diffusion.Transformer.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Diffusion.Transformer.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Diffusion.Transformer.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Diffusion.Transformer.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Diffusion.Transformer.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.Diffusion.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Diffusion.Size,
                num_head= self.hp.Diffusion.Transformer.Head,
                residual_conv_stack= self.hp.Diffusion.Transformer.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Diffusion.Transformer.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Diffusion.Transformer.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Diffusion.Transformer.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Diffusion.Transformer.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.Diffusion.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

        self.projection = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Audio_Codec_Size,
                kernel_size= 1), w_init_gain= 'zero'),
            )

        self.token_predictor = Token_Predictor(self.hp)

    def forward(
        self,
        noised_latents: torch.Tensor,
        encodings: torch.Tensor,
        f0s: torch.Tensor,
        lengths: torch.Tensor,        
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor,
        diffusion_steps: torch.Tensor,
        use_token_predictor: bool= False
        ):
        '''
        noised_latents: [Batch, Latent_d, Dec_t]
        encodings: [Batch, Enc_d, Dec_t]
        f0s: [Batch, Dec_t]
        diffusion_steps: [Batch]
        '''
        x = self.prenet(noised_latents)
        encodings = self.encoding_ffn(encodings)
        f0s = self.f0_ffn(f0s)
        prompts = self.prompt_ffn(prompts)
        diffusion_steps = self.step_ffn(diffusion_steps) # [Batch, Res_d, 1]

        x = x + encodings + f0s + diffusion_steps

        for block, prompt_block in zip(self.blocks, self.prompt_blocks):
            x = prompt_block(
                x= x,
                prompts= prompts,
                lengths= lengths,
                prompt_lengths= prompt_lengths
                )
            x = block(x, lengths)

        prediction_tokens = None
        if use_token_predictor:
            prediction_tokens = self.token_predictor(x)

        x = self.projection(x)

        return x, prediction_tokens

class Step_Embedding(torch.nn.Module):
    '''
    sinusoidal for float input
    '''
    def __init__(
        self,
        embedding_dim: int,
        ):
        super().__init__()
        assert embedding_dim % 2 == 0
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        div_term = math.log(10000) / (half_dim - 1)
        div_term = torch.exp(torch.arange(half_dim, dtype= torch.int64).float() * -div_term)

        self.register_buffer('div_term', div_term, persistent= False)

    def forward(self, x: torch.Tensor):
        x = x[:, None] * self.div_term[None]
        x = torch.cat([x.sin(), x.cos()], dim= 1)

        return x


class Token_Predictor(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.lstm = torch.nn.LSTM(
            input_size= self.hp.Audio_Codec_Size,
            hidden_size= self.hp.Diffusion.Token_Predictor.Size,
            num_layers= self.hp.Diffusion.Token_Predictor.LSTM.Stack,
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= self.hp.Diffusion.Token_Predictor.LSTM.Dropout_Rate,
            )

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Diffusion.Token_Predictor.Size,
            out_channels= self.hp.Tokens + 1,
            kernel_size= 1,
            ), w_init_gain= 'linear')
            
    def forward(
        self,
        encodings: torch.Tensor,
        ) -> torch.Tensor:
        '''
        features: [Batch, Feature_d, Feature_t], Spectrogram
        lengths: [Batch]
        '''
        encodings = encodings.permute(2, 0, 1)    # [Feature_t, Batch, Enc_d]
        
        self.lstm.flatten_parameters()
        encodings = self.lstm(encodings)[0] # [Feature_t, Batch, LSTM_d]
        
        predictions = self.projection(encodings.permute(1, 2, 0))
        predictions = torch.nn.functional.log_softmax(predictions, dim= 1)

        return predictions


@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x