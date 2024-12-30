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
from src.models.wavenet_modules import *

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

class PAEWave(nn.Module):
    def __init__(self, cfg):
        super(PAEWave, self).__init__()
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window
        
        # PAE parameters
        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False)
        
        # Build dilated convolution stack
        # WaveNet encoder parameters
        self.residual_channels = cfg.residual_channels
        self.dilation_channels = cfg.dilation_channels
        self.skip_channels = cfg.skip_channels
        self.layers = cfg.layers
        self.blocks = cfg.blocks
        self.dilation_channels = cfg.dilation_channels
        self.residual_channels = cfg.residual_channels
        self.skip_channels = cfg.skip_channels
        self.classes = self.input_channels
        self.kernel_size = cfg.kernel_size
        self.intermediate_channels = cfg.intermediate_channels
        #self.dtype = dtype #this was in the original code

        # build model
        # TODO: maybe add normalization

        # Initial convolution to residual_channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channels, out_channels=self.residual_channels, kernel_size=1, padding='same')

        # WaveNet Encoder Stack
        self.dilated_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channels, out_channels=self.residual_channels,kernel_size=1) 

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(self.kernel_size - 1) * new_dilation + 1,
                                                        num_channels=self.residual_channels, dilation=new_dilation ))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=self.residual_channels,out_channels=self.dilation_channels,
                                                    kernel_size=self.kernel_size, padding='same'))

                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels, out_channels=self.dilation_channels,
                                                 kernel_size=self.kernel_size,padding='same'))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,out_channels=self.residual_channels,kernel_size=1))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,out_channels=self.skip_channels,kernel_size=1))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        # not sure if we need 2 end_conv layers
        self.end_conv_1 = nn.Conv1d(in_channels=self.skip_channels, out_channels=self.intermediate_channels, kernel_size=1, bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=self.intermediate_channels, out_channels=self.embedding_channels, kernel_size=1, bias=True)

        self.receptive_field = receptive_field
        
        # PAE components
        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))



        # TODO: revisit and redefine the entire Decoder Architecture
        self.end_deconv = nn.Conv1d(in_channels=self.residual_channels, out_channels=self.input_channels,kernel_size=1)

        # Inverse WaveNet Decoder Stack
        self.inv_dilations = []
        self.inv_dilated_queues = []
        self.inv_filter_convs = nn.ModuleList()
        self.inv_gate_convs = nn.ModuleList()
        self.inv_residual_convs = nn.ModuleList()
        self.inv_skip_convs = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        

        # Build inverse dilated convolution stack (in reverse order)
        for b in range(self.blocks):
            new_dilation = 2**(self.layers-1)  # Start with largest dilation
            
            for i in range(self.layers-1, -1, -1):
                self.inv_dilations.append((new_dilation,init_dilation))

                # inverse dilated queues for fast generation
                self.inv_dilated_queues.append(DilatedQueue(max_length=(self.kernel_size - 1) * new_dilation + 1,
                                                        num_channels=self.residual_channels, dilation=new_dilation ))

                # Inverse dilated convolutions
                self.inv_filter_convs.append(nn.ConvTranspose1d(in_channels=self.dilation_channels, out_channels=self.residual_channels,
                                                      kernel_size=self.kernel_size, padding='same'))

                self.inv_gate_convs.append(nn.ConvTranspose1d(in_channels=self.dilation_channels,out_channels=self.residual_channels,
                                                    kernel_size=self.kernel_size, padding='same'))

                # 1x1 convolution for residual connection
                self.inv_residual_convs.append(nn.ConvTranspose1d(in_channels=self.residual_channels, out_channels=self.dilation_channels,kernel_size=1 ))

                 # 1x1 convolution for inverse skip connection (does this make sense??)
                self.inv_skip_convs.append(nn.ConvTranspose1d(in_channels=self.skip_channels, out_channels=self.dilation_channels,kernel_size=1 ))
                
                init_dilation = new_dilation
                new_dilation //= 2
        
        # Final output convolutions
        self.start_deconv_1 = nn.ConvTranspose1d(in_channels=self.embedding_channels,out_channels=self.intermediate_channels,kernel_size=1,bias=True)
        
        self.start_deconv_2 = nn.ConvTranspose1d(in_channels=self.intermediate_channels,out_channels=self.skip_channels, kernel_size=1,bias=True)
        
    

    def wavenet(self,x, dilation_func):
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)
            
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                 s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x
    
    def wavenet_inverse_dilate(self,input,dilation,init_dilation,i):
        #x = invert_dilate(input,dilation,init_dilation)
        return input

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size, dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def de_wavenet(self, x, dilation_func):
        x = self.start_deconv_1(x)
        x = F.relu(x)
        x = F.relu(self.start_deconv_2(x))
        
        skip = 0
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            
            (dilation, init_dilation) = self.inv_dilations[i]
            print(dilation,init_dilation)
            residual = dilation_func(x, dilation, init_dilation, i)
            print(residual.shape)
            # dilated convolution
            filter = self.inv_filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.inv_gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                s = dilate(x, 1, init_dilation=dilation)
            s = self.inv_skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.inv_residual_convs[i](x)
            x = x + residual

        x = F.relu(skip)
        x = self.end_deconv(x)
        print(x.shape)
        return x

    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        # Add small epsilon to prevent division by zero
        power_sum = torch.sum(power, dim=dim)
        eps = 1e-9
        
        # Safe division
        freq = torch.sum(self.freqs * power, dim=dim) / (power_sum + eps)
        
        # Safe sqrt for amplitude
        amp = 2 * torch.sqrt(power_sum + eps) / self.time_range
        
        # DC offset remains the same
        offset = rfft.real[:,:,0] / self.time_range

        return freq, amp, offset

    
    def forward(self, x):
        # Reshape input
        print(type(x),x)
        y = x.reshape(x.shape[0], self.input_channels, self.time_range)

        # WaveNet encoder
        y = self.wavenet(x, dilation_func=self.wavenet_dilate)
        latent = y
        # Extract frequency components
        f, a, b = self.FFT(y, dim=2)

        # Phase calculation
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi
        # Parameters
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)

        params = [p, f, a, b]
        # Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        signal = y
    
        # Signal Reconstruction using inverse WaveNet
        y = self.de_wavenet(y,dilation_func=self.wavenet_inverse_dilate)
        y = y.reshape(y.shape[0], self.input_channels*self.time_range)
       
        return y, latent, signal, params

class PAEDeep(nn.Module):
    """
    The PAE model adapted to accept 1D input
    """
    def __init__(self, cfg):
        super(PAEDeep, self).__init__()
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels1 = cfg.intermediate_channels1 # int(self.input_channels/3)
        intermediate_channels2 = cfg.intermediate_channels2
        self.conv1 = nn.Conv1d(self.input_channels, intermediate_channels1, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        #self.norm1 = LN_v2(self.time_range)
        self.conv2 = nn.Conv1d(intermediate_channels1, intermediate_channels2, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=2*cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(intermediate_channels2, self.embedding_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=4*cfg.dilation, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        self.deconv1 = nn.Conv1d(self.embedding_channels, intermediate_channels2, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=4*cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        #self.denorm1 = LN_v2(self.time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels2, intermediate_channels1, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=2*cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        self.deconv3 = nn.Conv1d(intermediate_channels1, self.input_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=cfg.dilation, groups=1, bias=True, padding_mode='zeros')
        
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
        #y = self.norm1(y)
        y = F.tanh(y)

        
        y = self.conv2(y)
        y = F.tanh(y)
        y = self.conv3(y)
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
        #y = self.denorm1(y)
        y = F.tanh(y)

        y = self.deconv2(y)
        y = F.tanh(y)
        y = self.deconv3(y)
        
        y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params
    
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
