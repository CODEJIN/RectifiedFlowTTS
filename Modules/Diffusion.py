import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from torchdiffeq import odeint

from .Layer import Conv_Init, Lambda

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
        network_output = self.network(
            noised_latents= noised_latents,
            encodings= encodings,
            f0s= f0s,
            diffusion_steps= cosmap_schedule_times[:, 0, 0] # [Batch]
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

        return flows, prediction_flows, noises, prediction_noises
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
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
            network_output = self.network(
                noised_latents= noised_latents,
                encodings= encodings,
                f0s= f0s,
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

        self.residual_blocks = torch.nn.ModuleList([
            Residual_Block(
                in_channels= self.hp.Diffusion.Size,
                kernel_size= self.hp.Diffusion.Kernel_Size,
                encoding_channels= self.hp.Encoder.Size
                )
            for _ in range(self.hp.Diffusion.Stack)
            ])
        
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

    def forward(
        self,
        noised_latents: torch.Tensor,
        encodings: torch.Tensor,
        f0s: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        '''
        noised_latents: [Batch, Latent_d, Dec_t]
        encodings: [Batch, Latent_d, Dec_t]
        f0s: [Batch, Dec_t]
        diffusion_steps: [Batch]
        '''
        x = self.prenet(noised_latents)
        f0s = f0s[:, None, :]   # [Batch, 1, Dec_t]
        diffusion_steps = self.step_ffn(diffusion_steps) # [Batch, Res_d, 1]

        skips_list = []
        for residual_block in self.residual_blocks:
            x, skips = residual_block(
                x= x,
                encodings= encodings,
                f0s= f0s,
                diffusion_steps= diffusion_steps
                )
            skips_list.append(skips)

        x = torch.stack(skips_list, dim= 0).sum(dim= 0) / math.sqrt(self.hp.Diffusion.Stack)
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

class Residual_Block(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        encoding_channels: int
        ):
        super().__init__()
        self.in_channels = in_channels
        
        self.encoding = torch.nn.Conv1d(
            in_channels= encoding_channels,
            out_channels= in_channels * 2,
            kernel_size= 1
            )
        self.f0 = torch.nn.Conv1d(
            in_channels= 1,
            out_channels= in_channels * 2,
            kernel_size= 1
            )

        self.diffusion_step = torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= in_channels,
            kernel_size= 1
            )

        self.conv = torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= in_channels * 2,
            kernel_size= kernel_size,
            padding= kernel_size // 2
            )

        self.projection = torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= in_channels * 2,
            kernel_size= 1
            )

    def forward(
        self,
        x: torch.FloatTensor,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
        diffusion_steps: torch.FloatTensor
        ):
        residuals = x

        encodings = self.encoding(encodings)
        f0s = self.f0(f0s)
        diffusion_steps = self.diffusion_step(diffusion_steps)

        x = self.conv(x + diffusion_steps) + encodings + f0s
        x = Fused_Gate(x)

        x = self.projection(x)
        x, skips = x.chunk(chunks= 2, dim= 1)

        return (x + residuals) / math.sqrt(2.0), skips

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x