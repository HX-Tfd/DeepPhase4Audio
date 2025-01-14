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

class PAEWave(nn.Module):
    def __init__(self, cfg):
        super(PAEWave, self).__init__()

        seed = 42 # TODO change this generic seed that everyone uses!!
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window
 
        # PAE parameters
        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False)
        
        # WaveNet encoder parameters
        self.residual_channels = cfg.residual_channels
        self.dilation_channels = cfg.dilation_channels
        self.skip_channels = cfg.skip_channels
        self.layers = cfg.layers
        self.blocks = cfg.blocks
        self.dilation_channels = cfg.dilation_channels
        self.residual_channels = cfg.residual_channels
        self.skip_channels = cfg.skip_channels
        self.classes = cfg.classes
        self.kernel_size = cfg.kernel_size
        self.intermediate_channels = cfg.intermediate_channels
        #self.dtype = dtype #this was in the original code

        # WaveNet Encoder Stack
        self.dilated_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        receptive_field = 1
        init_dilation = 1
        dilation_power = 1

        self.dilations = []
        self.dilated_queues = []

        # WaveNet Dencoder Stack
        self.decoder_dilated = nn.ModuleList()
        self.decoder_residual = nn.ModuleList()
        self.decoder_skip = nn.ModuleList()

        # ☆☆ Should we add mu law encoding to turn input into one hot vectors of size 256 ? Tried - but makes the latent space representation inconsistent with the periodic representation

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channels, out_channels=self.residual_channels,kernel_size=1) 

        # TODO: maybe add normalization

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

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
                dilation_power += 1

        # not sure if we need 2 end_conv layers
        self.end_conv_1 = nn.Conv1d(in_channels=self.skip_channels, out_channels=self.intermediate_channels, kernel_size=1, bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=self.intermediate_channels, out_channels=self.embedding_channels, kernel_size=1, bias=True)

        print(f"dilations for encoding: {new_dilation}")
        # ☆☆ then project wavenet to the embedding space in which we want to find the phase manifolds of size self.embedding_channels -- not sure, that might b redundant
        """
        self.to_embedding = nn.Sequential(
            nn.Conv1d(self.classes, self.intermediate_channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.intermediate_channels, self.embedding_channels, 1)
        )"""
        self.receptive_field = receptive_field
        
        # PAE components
        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        # ☆☆  if we did ewxtra projection - project back
        """
        self.from_embedding = nn.Sequential(
            nn.Conv1d(self.embedding_channels, self.intermediate_channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.intermediate_channels, self.classes, 1)
        )"""

        """
        DECODER - TRY 3
        """
        
        # Initialize improved decoder components - idk shouldnt it echo how many it does the other way around ?
        dilations = [2**i for i in range(self.layers+1)][::-1]  # Reversed dilations tensor
        print(f"dilations tensor for decoder : {dilations}")
    
        # Decoder initial projection
        self.decoder_initial = nn.ConvTranspose1d(self.embedding_channels, self.residual_channels, 1,bias=True)

        # Dilated convolution layers
        self.decoder_dilated = nn.ModuleList([
            nn.ConvTranspose1d(
                self.residual_channels,
                self.dilation_channels * 2,
                kernel_size=3,
                bias=True,
                padding=dilation if isinstance(dilation, int) else dilation[0],
                dilation=dilation if isinstance(dilation, int) else dilation[0]

            ) for dilation in self.dilations
        ])
        
        # Residual connections
        self.decoder_residual = nn.ModuleList([
            nn.ConvTranspose1d(
                self.dilation_channels,
                self.residual_channels,
                kernel_size=1,
                bias=True
            ) for _ in self.dilations
        ])
        
        # Skip connections
        self.decoder_skip = nn.ModuleList([
            nn.ConvTranspose1d(
                self.dilation_channels,
                self.skip_channels,
                kernel_size=1,
                bias=True
            ) for _ in self.dilations
        ])
        
        # Final layers with transposed convolutions
        self.decoder_final = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.skip_channels,
                self.intermediate_channels,
                kernel_size=1,
                bias=True
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.intermediate_channels,
                self.input_channels,
                kernel_size=1,
                bias=True
            )
        )

        """
        DECODER - TRY 1
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
        """

        """
        DECODER - TRY 2 
        dilations = [2**i for i in range(9)][::-1] # Reversed dilations tensor
        print(f"dilations tensor for decoder : {dilations}")
        # Decoder initial projection
        self.decoder_initial = nn.Conv1d(self.embedding_channels, self.residual_channels, 1)
        for dilation in dilations:
        self.decoder_dilated.append(
            nn.Conv1d(
            self.residual_channels,
            self.dilation_channels * 2,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
            )
        )
        self.decoder_residual.append(
            nn.Conv1d(
            self.dilation_channels,
            self.residual_channels,
            1
            )
        )
        self.decoder_skip.append(
            nn.Conv1d(
            self.dilation_channels,
            self.skip_channels,
            1
            )
        )
        # Decoder final layers
        self.decoder_final = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.skip_channels, self.intermediate_channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.intermediate_channels, self.input_channels, 1),
            )
        """

    def wavenet(self,x, dilation_func):
        x = self.start_conv(x)
        skip = None

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
            if skip is None:
                skip = s
            else:
                skip = skip[:, :, -s.size(2):]  # Trim skip to match s
                s = s[:, :, -skip.size(2):]     # Trim s to match skip
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual
            # x = x + residual[:, :, (self.kernel_size - 1):] -- original shape

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x
    
    """
    NOT USED
    def wavenet_inverse_dilate(self,input,dilation,init_dilation,i):
        #x = invert_dilate(input,dilation,init_dilation)
        return input
    """

    """
    NOT USED
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
    """

    

    def decode(self, x):
        x = self.decoder_initial(x)
        
        # Storage for skip connections
        skip_connections = []
        
        # Process through dilated layers in reverse order
        for i, (dilated, residual, skip) in enumerate(zip(
            self.decoder_dilated,
            self.decoder_residual,
            self.decoder_skip
        )):
            # Store residual
            residual_x = x
            
            # Dilated convolution
            x = dilated(x)
            
            # Gating mechanism
            tanh_out, sigmoid_out = torch.chunk(x, 2, dim=1)
            x = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
            #x = tanh_out * sigmoid_out # maybe dont do the gating mechanism
            # Split for residual and skip connections
            # For skip connection
            s = skip(x)
            skip_connections.append(s)
            
            # For residual connection
            x = residual(x)
            
            # Add residual
            x = (x + residual_x) * 0.707  # Scale by 1/√2 for stability
        
        # Combine skip connections
        x = torch.stack(skip_connections).sum(dim=0)
        
        # Final layers
        x = self.decoder_final(x)

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
        # Input shape: [B, input_channels, T]
        y = x.reshape(x.shape[0], self.input_channels, self.time_range)

        #print(f"y before encoding : {y.shape}")
    
        # μ-law encoding necessary ?
    
        # WaveNet encoder
        y = self.wavenet(y, dilation_func=self.wavenet_dilate)
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
    
        # Decode through inverse WaveNet
        y = self.decode(y)  # [B, classes, T]
    
        # μ-law decode back to waveform ?
    
        # Final reshape
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
    

class PAElla(nn.Module):
    """
    The PAE model adapted to accept 1D input
    """
    seed = 42 # TODO change this generic seed that everyone uses!!
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    def __init__(self, cfg):
        super(PAElla, self).__init__()
        self.input_channels = cfg.input_channels
        self.embedding_channels = cfg.embedding_channels
        self.time_range = cfg.time_range
        self.window = cfg.window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * self.time_range / self.window, requires_grad=False) #Remove DC frequency

        self.first_layer = nn.ModuleList()
        self.second_layer = nn.ModuleList()
        self.embedding_channels = cfg.embedding_channels
        self.depth = int(math.log2(self.embedding_channels))
        self.deconv_layers =[]

        for i in range(self.embedding_channels):
            conv1 =nn.Conv1d(1, 2, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=(2**i), groups=1, bias=True, padding_mode='zeros')
            conv2 = nn.Conv1d(2, 1, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=(2**i), groups=1, bias=True, padding_mode='zeros')
            self.first_layer.append(conv1)
            self.second_layer.append(conv2)


        for i in range(self.depth):
            deconv_layer_i = nn.ModuleList()
            for j in range(2**(self.depth-i-1)):
                conv = nn.Conv1d(2,1, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros')
                deconv_layer_i.append(conv)
            self.deconv_layers.append(deconv_layer_i)


        self.final_conv = nn.Conv1d(2, self.input_channels, kernel_size=cfg.kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for _ in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))

        
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
        y2 = torch.zeros(y.shape[0],self.embedding_channels, self.time_range)
        y2 = []

        for i in range(self.embedding_channels):
            y1 = self.first_layer[i](y)
            y2.append(self.second_layer[i](y1))

        # Concatenate the outputs of all second_layer convolutions along the channel dimension
        y = torch.cat(y2, dim=1)

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
   
        for j in range(self.depth):
            y1 = []#torch.zeros(y.shape[0], 4, self.time_range)
            for i in range(2**(self.depth-j-1)):
                y1.append(self.deconv_layers[j][i](y[:,2*i:(2*i+2),:]))
            y1 = torch.cat(y1, dim=1)
            y = y1
        
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
