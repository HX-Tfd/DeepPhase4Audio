"""
This code is adapted from 
https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/Library/Utility.py 
https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE 
"""

from functools import partial
import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F    

  
from .PAE import PAE
from .modules import LN_v2, Snake1d, WNConv1d, WNConvTranspose1d, activation
from .vq import ResidualVectorQuantize
from .latent_combinator import CACombiner, MLPCombiner

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VQ_AE(PAE):
    """
    The PAE model with (Residual) Vector Quantization
    """
    def __init__(self, cfg):
        super(VQ_AE, self).__init__(cfg)
        # PAE configs
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window
        self.use_fft_mlp = cfg.use_fft_mlp
        self.vq_only = cfg.vq_only
        
        # VQ configs
        self.quantizer = ResidualVectorQuantize(**cfg.vq.to_dict())
        
        # PAE modules
        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency
        intermediate_channels = cfg.intermediate_channels # int(self.input_channels/3)
        self.activation = activation(cfg.activation)
            
        self.enc_dilation_rates = cfg.enc_dilation_rates
        self.dec_dilation_rates = cfg.dec_dilation_rates
        self.enc_kernel_sizes = cfg.enc_kernel_sizes
        self.dec_kernel_sizes = cfg.dec_kernel_sizes
        self.enc_layers = len(self.enc_dilation_rates)
        self.dec_layers = len(self.dec_dilation_rates)
        enc_modules, dec_modules = [], []
        for i in range(self.enc_layers):
            if i == 0:
                enc_modules.append(nn.Conv1d(self.input_channels, intermediate_channels, kernel_size=self.enc_kernel_sizes[i], stride=1, padding='same', dilation=self.enc_dilation_rates[i], groups=1, bias=True, padding_mode='zeros'))
                enc_modules.append(LN_v2(self.time_range))
                enc_modules.append(self.activation)
            elif i == self.enc_layers - 1:
                enc_modules.append(nn.Conv1d(intermediate_channels, self.embedding_channels, kernel_size=self.enc_kernel_sizes[i], stride=1, padding='same', dilation=self.enc_dilation_rates[i], groups=1, bias=True, padding_mode='zeros'))
            else:
                enc_modules.append(nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=self.enc_kernel_sizes[i], stride=1, padding='same', dilation=self.enc_dilation_rates[i], groups=1, bias=True, padding_mode='zeros'))
                enc_modules.append(LN_v2(self.time_range))
                enc_modules.append(self.activation)
                
        for i in range(self.dec_layers):
            if i == 0:
                dec_modules.append(nn.Conv1d(self.embedding_channels, intermediate_channels, kernel_size=self.dec_kernel_sizes[i], stride=1, padding='same', dilation=self.dec_dilation_rates[i], groups=1, bias=True, padding_mode='zeros'))
                dec_modules.append(LN_v2(self.time_range))
                dec_modules.append(self.activation)
            elif i == self.enc_layers - 1:
                dec_modules.append(nn.Conv1d(intermediate_channels, self.input_channels, kernel_size=self.dec_kernel_sizes[i], stride=1, padding='same', dilation=self.dec_dilation_rates[i], groups=1, bias=True, padding_mode='zeros'))
            else:
                dec_modules.append(nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=self.dec_kernel_sizes[i], stride=1, padding='same', dilation=self.dec_dilation_rates[i], groups=1, bias=True, padding_mode='zeros'))
                dec_modules.append(LN_v2(self.time_range))
                dec_modules.append(self.activation)
        
        self.encoder = nn.Sequential(*enc_modules)
        self.decoder = nn.Sequential(*dec_modules)
        
        # process latents by separate FFNs because the features and control signals are not in the same space (?)
        self.phase_ffn = nn.Sequential(
            nn.LayerNorm(self.embedding_channels),
            nn.Linear(self.embedding_channels, self.embedding_channels * 2),
            self.activation,
            nn.LayerNorm(self.embedding_channels*2),
            nn.Linear(self.embedding_channels * 2, self.embedding_channels),
            self.activation,
        )
        self.vq_ffn = nn.Sequential(
            nn.LayerNorm(self.embedding_channels),
            nn.Linear(self.embedding_channels, self.embedding_channels * 2),
            self.activation,
            nn.LayerNorm(self.embedding_channels*2),
            nn.Linear(self.embedding_channels * 2, self.embedding_channels),
            self.activation,
        )
        
        
        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        if self.use_fft_mlp:
            fft_mlp_int_channels = 128
            in_length = self.time_range // 2
            self.fft_mlp = nn.Sequential(
                nn.Linear(in_length, in_length),
                nn.LeakyReLU(),
                nn.Linear(in_length, 1),
                nn.ELU()
            )

        if not self.vq_only:
            if cfg.combiner == 'xattn':
                self.latent_combinator = CACombiner(in_ch=self.embedding_channels,
                                                    key_ch=self.embedding_channels,
                                                    value_ch=self.embedding_channels,
                                                    n_heads=8
                                                    )
            elif cfg.combiner == 'concat':
                self.latent_combinator = MLPCombiner(channels=self.embedding_channels)
            else:
                raise NotImplementedError(f"combiner module {cfg.combiner} is not implemented!")
        
        
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] # Spectrum without DC component
        power = spectrum**2

        # Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        # Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        # Offset
        offset = rfft.real[:,:,0] / self.time_range # DC component

        return freq, amp, offset
        
    def encode(self, x):
        # Signal Embedding
        x = x.reshape(x.shape[0], self.input_channels, self.time_range)
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        x = x.reshape(x.shape[0], self.input_channels*self.time_range)
        return x

    def forward(self, x, n_quantizers: int = None):
        z = self.encode(x)
        z_pae = z 
        
        # rvq
        z_vq = self.vq_ffn(z.permute(0, 2, 1))
        z_vq, codes, latents, commitment_loss, codebook_loss = self.quantizer(z_vq.permute(0, 2, 1), n_quantizers) 
        z_comb = z_vq
        
        # disentangle phase 
        z_recon, phase_params = None, None
        if not self.vq_only:
            z_phase = self.phase_ffn(z.permute(0, 2, 1)).permute(0, 2, 1) # weird flex
            f, a, b = self.FFT(z_phase, dim=2) # Frequency, Amplitude, Offset

            p = torch.empty((z_phase.shape[0], self.embedding_channels), dtype=torch.float32, device=z_phase.device)
            for i in range(self.embedding_channels):
                v = self.fc[i](z_phase[:,i,:])
                p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

            p = p.unsqueeze(2)
            f = f.unsqueeze(2)
            a = a.unsqueeze(2)
            b = b.unsqueeze(2)
            
            phase_params = [p, f, a, b] # Save latent space parameters for returning
            z_recon = a * torch.sin(self.tpi * (f * self.args + p)) + b

            # combine discrete and continuous latent features
            z_comb = self.latent_combinator(z_recon, z_vq)
        
        # Signal Reconstruction
        x = self.decode(z_comb)
        
        return {
            "audio": x[..., :self.time_range],
            "z_pae": z_pae, 
            "z_vq": z_vq,
            "z_recon": z_recon,
            "phase_params": phase_params,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
    

if __name__ == "__main__":
    from src.utils.helpers import DotDict
    cfg = DotDict({
        'input_channels': 1,
        'embedding_channels': 16, # latent dim (= vq input dim)
        'intermediate_channels': 64,
        'time_range': 32000,
        'window': 2.0, 
        'use_fft_mlp': False,
        'enc_dilation_rates': [9, 3, 1],
        'enc_kernel_sizes': [7, 3, 3],
        'dec_dilation_rates': [1, 3, 9],
        'dec_kernel_sizes': [3, 3, 7],
        'activation': 'snake',
        'vq_only': False,
        'vq': {
            'input_dim': 16, # latent dim
            'n_codebooks': 9,
            'codebook_size': 128,
            'codebook_dim': 8,
            'quantizer_dropout': 0.0,
        },
        'combiner': 'concat'
    })
    
    model = VQ_AE(cfg)
    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params"
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
        
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))
    
    x = torch.rand(16, 32000, 1)
    d = model(x)
    x_recon = d["audio"] # B, L
    z_pae = d["z_pae"] # B, emb_ch, L
    z_vq = d["z_vq"]   # B, emb_ch, L
    z_recon = d["z_recon"]
    phase_params = d["phase_params"]
    codes = d["codes"] # B, n_codebooks, L
    latents = d["latents"] # B, n_codebooks x codebook_dim, L
    vq_comm_loss = d[ "vq/commitment_loss"]
    vq_cb_loss = d["vq/codebook_loss"]
    
    p, f, a, b = phase_params
    print(x_recon.shape, z_pae.shape, z_vq.shape, codes.shape, latents.shape)
    print(z_recon.shape, p.shape, f.shape, a.shape, b.shape)
    print(torch.cat((z_recon, z_vq), dim=1).shape)