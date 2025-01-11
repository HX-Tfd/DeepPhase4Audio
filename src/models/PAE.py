"""
This code is adapted from 
https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/Library/Utility.py 
https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE 
"""

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

from src.utils.model_utils import count_model_params

from .modules import LN_v2, Snake1d, positional_encoding

class PAE(nn.Module):
    """
     The PAE model from the original paper.
     Assumptions:
        input (motion curve) shape: (B x) D x N, where D = dof, The temporal data is divided into overlapping windows of length N
        with corresponding centered time window T.
    """

    def __init__(self, cfg):
        super(PAE, self).__init__()
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = int(self.input_channels/3)
        
        self.conv1 = nn.Conv1d(self.input_channels, intermediate_channels, self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = LN_v2(self.time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, self.embedding_channels, self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        self.deconv1 = nn.Conv1d(self.embedding_channels, intermediate_channels, self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = LN_v2(self.time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, self.input_channels, self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params
    
    

class PAEInputFlattened_S(nn.Module):
    """
    The PAE model adapted to accept 1D input
    """
    def __init__(self, cfg):
        super(PAEInputFlattened_S, self).__init__()
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = cfg.intermediate_channels # int(self.input_channels/3)
        
        self.conv1 = nn.Conv1d(self.input_channels, intermediate_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = LN_v2(self.time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, self.embedding_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        self.deconv1 = nn.Conv1d(self.embedding_channels, intermediate_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = LN_v2(self.time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, self.input_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        
        fft_mlp_int_channels = 128
        in_length = self.time_range // 2
        self.fft_mlp = nn.Sequential(
            nn.Linear(in_length, in_length),
            nn.LeakyReLU(),
            nn.Linear(in_length, 1),
            nn.ELU()
        )

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        # freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        freq = self.fft_mlp(power).squeeze(-1)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range # DC component

        return freq, amp, offset
    

    def forward(self, x): # B, 2, L
        y = x[:, 0, :] #+ positional_encoding(pos=x[:, 1, :])

        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)     

        latent = y #Save latent for returning
        # print("latent shape:", y.shape)

        # FFT branch: Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        # Phase branch
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params
    
    
class AE(PAE):
    """
    The PAE model with the latent construct removed
    """
    def __init__(self, cfg):
        super(AE, self).__init__(cfg)
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = cfg.intermediate_channels # int(self.input_channels/3)
        
        self.conv1 = nn.Conv1d(self.input_channels, intermediate_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = LN_v2(self.time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, self.embedding_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        self.deconv1 = nn.Conv1d(self.embedding_channels, intermediate_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = LN_v2(self.time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, self.input_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')


    def forward(self, x):
        y = x

        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)

        latent = y #Save latent for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent
    
    
class PAEInputFlattened(nn.Module):
    """
    The PAE model adapted to accept 1D input
    """
    def __init__(self, cfg):
        super(PAEInputFlattened, self).__init__()
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window
        self.use_fft_mlp = cfg.use_fft_mlp
        self.activation = Snake1d() # nn.ELU()

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = cfg.intermediate_channels # int(self.input_channels/3)
        
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

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        if self.use_fft_mlp:
            freq = self.fft_mlp(power).squeeze(-1)
        else:
            freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range # DC component

        return freq, amp, offset
    

    def forward(self, x): # B, 2, L
        y = x[:, 0, :] #+ positional_encoding(pos=x[:, 1, :])

        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.encoder(y)

        latent = y #Save latent for returning
        # print("latent shape:", y.shape)

        # FFT branch: Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        # Phase branch
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.decoder(y)

        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params
    


if __name__ == "__main__":
    from src.utils.helpers import DotDict
    cfg = DotDict({
        'input_channels': 1,
        'embedding_channels': 15,
        'intermediate_channels': 16,
        'time_range': 32000,
        'window': 2.0,
        'fft_mlp': False,
        'enc_dilation_rates': [9, 7, 5],
        'enc_kernel_sizes': [51, 21, 7],
        'dec_dilation_rates': [5, 7, 9],
        'dec_kernel_sizes': [7, 21, 51]
    })
    model = PAEInputFlattened(cfg)
    print(model)
    print("trainable model parameters: ", count_model_params(model))
    
    x = torch.rand(32, 2, 32000)
    y, _, _, _ = model(x)
    print(y)
    
    # ae_model = AE(cfg)
    # x = torch.rand(16, 32000, 1)
    # y, _ = ae_model(x)


