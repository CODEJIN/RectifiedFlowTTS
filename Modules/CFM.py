import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm

from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

from .Layer import Conv_Init, Lambda, FFT_Block, Norm_Type

# https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py

class CFM(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.network = Network(
            hyper_parameters= self.hp
            )

        if self.hp.CFM.Scheduler.upper() == 'Uniform'.upper():
            self.timestep_scheduler = lambda x: x
        elif self.hp.CFM.Scheduler.upper() == 'Cosmap'.upper():
            self.timestep_scheduler = lambda x: 1.0 - (1 / (torch.tan(torch.pi / 2.0 * x) + 1))
        else:
            raise NotImplementedError(f'Unsupported CFM scheduler: {self.hp.CFM.Scheduler}')

    def forward(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
        mels: torch.FloatTensor,
        lengths: torch.IntTensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor
        ):
        '''
        encodings: [Batch, Enc_d, Dec_t]
        mels: [Batch, Mel_d, Mel_t]
        '''        
        if self.hp.CFM.Use_CFG:
            cfg_filters = (torch.rand(
                encodings.size(0),
                device= encodings.device
                )[:, None, None] > self.hp.Train.CFG_Alpha).float()  # [Batch, 1, 1]            
            encodings = encodings * cfg_filters
            f0s = f0s * cfg_filters[:, 0, :]
            prompts = prompts * cfg_filters
        
        timesteps = self.timestep_scheduler(
            torch.rand(encodings.size(0), device= encodings.device)
            )[:, None, None]

        noises = torch.randn_like(mels)  # [Batch, Mel_d, Mel_t]

        noised_mels = timesteps * mels + (1.0 - timesteps) * noises
        flows = mels - noises

        # predict flow
        network_outputs = self.network.forward(
            noised_mels= noised_mels,
            encodings= encodings,
            f0s= f0s,
            lengths= lengths,
            prompts= prompts,
            prompt_lengths= prompt_lengths,
            timesteps= timesteps[:, 0, 0]  # [Batch]
            )
        prediction_flows = network_outputs

        return flows, prediction_flows
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
        lengths: torch.IntTensor,
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor,
        steps: int,
        cfg_guidance_scale: Optional[float]= 4.0
        ):
        noises = torch.randn(
            encodings.size(0), self.hp.Sound.N_Mel, encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, Mel_d, Mel_t]
        
        def ode_func(timesteps: torch.Tensor, noised_mels: torch.Tensor):
            timesteps = timesteps[None].expand(noised_mels.size(0))  # [Batch]

            # predict flow
            network_outputs = self.network.forward(
                noised_mels= noised_mels,
                encodings= encodings,
                f0s= f0s,
                lengths= lengths,
                prompts= prompts,
                prompt_lengths= prompt_lengths,
                timesteps= timesteps
                )
            if self.hp.CFM.Use_CFG:
                network_outputs_without_condition = self.network.forward(
                    noised_mels= noised_mels,
                    encodings= torch.zeros_like(encodings),
                    f0s= torch.zeros_like(f0s),
                    lengths= lengths,
                    prompts= torch.zeros_like(prompts),
                    prompt_lengths= prompt_lengths,
                    timesteps= timesteps
                    )
                network_outputs = network_outputs + cfg_guidance_scale * (network_outputs - network_outputs_without_condition)
            
            prediction_flows = network_outputs

            return prediction_flows
        
        mels = odeint(
            func= ode_func,
            y0= noises,
            t= self.timestep_scheduler(torch.linspace(0.0, 1.0, steps, device= encodings.device)),
            atol= 1e-5,
            rtol= 1e-5,
            method= 'dopri5'
            )[-1]

        return mels

class Network(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Sound.N_Mel,
                out_channels= self.hp.CFM.Size,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )
        
        self.encoding_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Sound.N_Mel,
                out_channels= self.hp.CFM.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.CFM.Size * 4,
                out_channels= self.hp.CFM.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )

        self.f0_ffn = torch.nn.Sequential(
            Lambda(lambda x: x[:, :, None]),
            Conv_Init(torch.nn.Linear(
                in_features= 1,
                out_features= self.hp.CFM.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.CFM.Size * 4,
                out_features= self.hp.CFM.Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x.mT)
            )
        
        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.CFM.Size
                ),            
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.CFM.Size,
                out_features= self.hp.CFM.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.CFM.Size * 4,
                out_features= self.hp.CFM.Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x[:, :, None])
            )

        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.CFM.Size,
                num_head= self.hp.CFM.Transformer.Head,
                ffn_kernel_size= self.hp.CFM.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.CFM.Transformer.FFN.Dropout_Rate,
                norm_type= Norm_Type.Conditional_LayerNorm,
                condition_channels= self.hp.Encoder.Size,
                )
            for _ in range(self.hp.CFM.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

        self.projection = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.CFM.Size,
                out_channels= self.hp.CFM.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.CFM.Size,
                out_channels= self.hp.Sound.N_Mel,
                kernel_size= 1), w_init_gain= 'zero'),
            )

    def forward(
        self,
        noised_mels: torch.Tensor,
        encodings: torch.Tensor,
        f0s: torch.Tensor,
        lengths: torch.Tensor,        
        prompts: torch.FloatTensor,
        prompt_lengths: torch.IntTensor,
        timesteps: torch.Tensor
        ):
        '''
        noised_mels: [Batch, Mel_d, Dec_t]
        encodings: [Batch, Enc_d, Dec_t]
        f0s: [Batch, Dec_t]
        timesteps: [Batch]
        '''
        x = self.prenet(noised_mels)
        encodings = self.encoding_ffn(encodings)
        f0s = self.f0_ffn(f0s)
        timesteps = self.step_ffn(timesteps) # [Batch, Res_d, 1]

        x = x + encodings + f0s + timesteps

        for block in self.blocks:
            x = block(
                x= x,
                lengths= lengths,
                conditions= prompts,
                condition_lengths= prompt_lengths
                )

        x = self.projection(x)

        return x

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


@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x