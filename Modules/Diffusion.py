import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from torchdiffeq import odeint

from .Layer import Conv_Init, Lambda, FFT_Block, Positional_Encoding, Mask_Generate

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
        network_output = self.network.forward(
            noised_latents= noised_latents,
            encodings= encodings,
            f0s= f0s,
            diffusion_steps= cosmap_schedule_times[:, 0, 0], # [Batch]
            lengths= lengths
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
        lengths: torch.IntTensor,
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
                diffusion_steps= cosmap_schedule_times[:, 0, 0], # [Batch]
                lengths= lengths
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
                out_channels= self.hp.Diffusion.Initial_Size,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )

        self.f0_ffn = torch.nn.Sequential(
            Lambda(lambda x: x[:, :, None]),
            Conv_Init(torch.nn.Linear(
                in_features= 1,
                out_features= self.hp.Diffusion.Initial_Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Initial_Size * 4,
                out_features= self.hp.Diffusion.Initial_Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x.mT),
            )
        
        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.Diffusion.Initial_Size
                ),            
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Initial_Size,
                out_features= self.hp.Diffusion.Initial_Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Initial_Size * 4,
                out_features= self.hp.Diffusion.Initial_Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x[:, :, None])
            )

        self.encoder_blocks = torch.nn.ModuleList()
        self.decoder_blocks = torch.nn.ModuleList()

        '''
        if input tensor's shape is [Batch, Diffusion_d, 1000], stack is 5, unet initial size is 64.
        encoder_0 output: [Batch, 64, 1000]
        encoder_1 output: [Batch, 128, 250]
        encoder_2 output: [Batch, 256, 125]
        encoder_3 output: [Batch, 512, 65]
        encoder_4 output: [Batch, 1024, 33]

        transformer output: [Batch, 1024, 33]

        decoder_4 output: [Batch, 512, 66], last index must be cut.
        '''
        for stack_index in range(self.hp.Diffusion.UNet.Stack):
            in_channels = self.hp.Diffusion.Initial_Size * 2 ** stack_index
            out_channels = self.hp.Diffusion.Initial_Size * 2 ** (stack_index + 1)
            self.encoder_blocks.append(UNet_Block(
                in_channels= in_channels,
                out_channels= out_channels,
                kernel_size= self.hp.Diffusion.UNet.Kernel_Size,                
                ))
            self.decoder_blocks.append(UNet_Block(
                in_channels= out_channels,
                out_channels= in_channels,
                kernel_size= self.hp.Diffusion.UNet.Kernel_Size,
                use_upsample= stack_index > 0
                ))
            
        self.pool = torch.nn.MaxPool1d(kernel_size= 2, ceil_mode= True)

        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= out_channels
            )

        self.fft_blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= out_channels,
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
                in_channels= self.hp.Diffusion.Initial_Size,
                out_channels= self.hp.Diffusion.Initial_Size * 4,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Initial_Size * 4,
                out_channels= self.hp.Audio_Codec_Size,
                kernel_size= 1), w_init_gain= 'zero'),
            )

    def forward(
        self,
        noised_latents: torch.Tensor,
        encodings: torch.Tensor,
        f0s: torch.Tensor,
        diffusion_steps: torch.Tensor,
        lengths: torch.IntTensor
        ):
        '''
        noised_latents: [Batch, Latent_d, Dec_t]
        encodings: [Batch, Latent_d, Dec_t]
        f0s: [Batch, Dec_t]
        diffusion_steps: [Batch]
        '''
        x = self.prenet(noised_latents) # [Batch, Diffusion_d, Dec_t]
        f0s = self.f0_ffn(f0s)   # [Batch, Diffusion_d, Dec_t]
        diffusion_steps = self.step_ffn(diffusion_steps) # [Batch, Diffusion_d, 1]

        x = x + f0s + diffusion_steps

        encoding_skips = [] # if input tensor's length is 1000, inserted tensors' lengths are 1000, 500, 250, 125.
        length_skips = []
        for index, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x, lengths)
            if index < self.hp.Diffusion.UNet.Stack - 1:
                encoding_skips.append(x)
                length_skips.append(lengths)
                x = self.pool(x)
                lengths = torch.ceil(lengths.float() / 2).int()
        encoding_skips.append(torch.zeros_like(x))  # for decoder calc. 63.
        length_skips.append(lengths)  # for decoder calc. 63.

        x = x + self.positional_encoding(
            position_ids= torch.arange(x.size(2), device= x.device)[None]
            ).mT
        
        for block in self.fft_blocks:
            x = block(x, lengths)

        for decoder_block in reversed(self.decoder_blocks):
            encodings = encoding_skips.pop()
            lengths = length_skips.pop()
            x = decoder_block(x[:, :, :encodings.size(2)] + encodings, lengths) # 126, 250, 500, 1000, 1000

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

class UNet_Block(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_upsample: bool= False,
        ):
        super().__init__()
        self.use_upsample = use_upsample        
        
        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2
            ), w_init_gain= 'gelu')
        self.norm_0 = torch.nn.LayerNorm(out_channels)
        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= out_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2
            ), w_init_gain= 'gelu')
        self.norm_1 = torch.nn.LayerNorm(out_channels)

        self.gelu = torch.nn.GELU(approximate= 'tanh')

        if use_upsample:
            self.upsample = Conv_Init(torch.nn.ConvTranspose1d(
                in_channels= out_channels,
                out_channels= out_channels,
                kernel_size= 2,
                stride= 2
                ), w_init_gain= 'gelu')

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.IntTensor
        ):
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())    # [Batch, Time]
            float_masks = (~masks).unsqueeze(1).float()   # float mask, [Batch, 1, X_t]

        x = self.conv_0(x * float_masks)
        x = self.norm_0(x.mT).mT
        x = self.gelu(x)
        x = self.conv_1(x * float_masks)
        x = self.norm_1(x.mT).mT
        x = self.gelu(x)
        
        if self.use_upsample:
            x = self.upsample(x * float_masks)

        return x

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x