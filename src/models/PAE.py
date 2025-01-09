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

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


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
    
    
class MultiDilationConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation_rates):
        super(MultiDilationConv, self).__init__()
        self.paths = nn.ModuleList([
            nn.Conv1d(
                input_channels, output_channels, kernel_size, 
                stride=1, padding=(kernel_size - 1) // 2 * d, dilation=d, bias=True
            )
            for d in dilation_rates
        ])
        
    def forward(self, x):
        x = x.float()
        outputs = [path(x) for path in self.paths]
        combined = torch.cat(outputs, dim=1)  # concat along channel dimension
        return combined


class PAEInputFlattened(nn.Module):
    def __init__(self, cfg):
        super(PAEInputFlattened, self).__init__()
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0 * np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window / 2, self.window / 2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False)  # Remove DC frequency

        intermediate_channels = cfg.intermediate_channels

        #MultiDilationConv for embedding
        dilation_rates = [3, 17, 67, 199, 997]  # Specify dilation rates as needed
        self.multi_dilated_conv1 = MultiDilationConv(self.input_channels, intermediate_channels, cfg.kernel_size, dilation_rates)
        self.projection1 = nn.Conv1d(intermediate_channels * len(dilation_rates), intermediate_channels, kernel_size=1)
        self.norm1 = LN_v2(self.time_range)

        #Added extra convolution and normalization in encoder
        self.extra_conv1 = nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1)
        self.extra_norm1 = LN_v2(self.time_range)

        self.multi_dilated_conv2 = MultiDilationConv(intermediate_channels, self.embedding_channels, cfg.kernel_size, dilation_rates)
        self.projection2 = nn.Conv1d(self.embedding_channels * len(dilation_rates), self.embedding_channels, kernel_size=1)

        #Fully connected layers for phase calculation
        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        # Decoding layers
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(self.embedding_channels, intermediate_channels, kernel_size=k, padding=(k - 1) // 2) 
            for k in [3, 5, 7]
        ])
        self.deprojection1 = nn.Conv1d(intermediate_channels * len(self.multi_scale_conv), intermediate_channels, kernel_size=1)
        self.denorm1 = LN_v2(self.time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, self.input_channels, kernel_size=cfg.kernel_size, stride=1, 
                                 padding=(cfg.kernel_size - 1) // 2, dilation=1, bias=True)

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:, :, 0] / self.time_range  # DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.multi_dilated_conv1(y)  # Expand channels
        y = self.projection1(y)         # Reduce back to intermediate_channels
        y = self.norm1(y)
        y = F.elu(y)

        # Extra convolution in encoder
        y = self.extra_conv1(y)
        y = self.extra_norm1(y)
        y = F.elu(y)

        y = self.multi_dilated_conv2(y)  # Expand channels
        y = self.projection2(y)          # Reduce back to embedding_channels
        latent = y  # Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:, i, :])
            p[:, i] = torch.atan2(v[:, 1], v[:, 0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b]  # Save parameters for returning

        # Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y  # Save signal for returning

        # Multi-scale feature extraction in decoder
        recon_features = [conv(y) for conv in self.multi_scale_conv]
        y = torch.cat(recon_features, dim=1)  # Concatenate multi-scale features

        y = self.deprojection1(y)  # Reduce back to intermediate_channels
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)  # Final reconstruction

        y = y.reshape(y.shape[0], self.input_channels * self.time_range)

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
    

if __name__ == "__main__":
    from src.utils.helpers import DotDict
    cfg = DotDict({
        'input_channels': 1,
        'embedding_channels': 5,
        'intermediate_channels': 16,
        'kernel_size': 11,
        'dilation': 5,
        'time_range': 32000,
        'window': 2.0
    })
    model = PAEInputFlattened(cfg)
    print(model)
    
    x = torch.rand(16, 32000, 1)
    y, _, _, _ = model(x)
    print(y)
    
    ae_model = AE(cfg)
    x = torch.rand(16, 32000, 1)
    y, _ = ae_model(x)


