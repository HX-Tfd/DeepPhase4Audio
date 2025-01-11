import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


def invert_dilate(x, orig_n, init_dilation=1):
    """
    Invert the dilation process.
    
    :param x: Dilated tensor of size (dilation, C, L*N / dilation)
    :param orig_n: Original N value before dilation
    :param init_dilation: Initial dilation factor, default is 1
    :return: The original tensor of size (N, C, L)
    """

    [d, c, l] = x.size()
    dilation_factor = d * init_dilation
    if dilation_factor == 1:
        return x

    new_l = l
    x = x.permute(1, 2, 0).contiguous()  # (d, c, l) -> (c, l, d)
    x = x.view(c, new_l, orig_n)
    x = x.permute(2, 0, 1).contiguous()  # (c, new_l, orig_n) -> (orig_n, c, new_l)

    return x

def mu_law_encode(x):
    """Convert audio to one-hot mu-law encoded vectors"""
    mu = 255  # μ-law compression (256 levels)
    
    # Ensure input is in [-1, 1]
    x = torch.clamp(x, -1, 1)
    
    # μ-law compression
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu, device=x.device, dtype=x.dtype))
    
    # Quantize to 256 levels
    x_quant = ((x_mu + 1) / 2 * mu).long()
    
    # Create one-hot encoding
    x_onehot = torch.nn.functional.one_hot(x_quant, num_classes=mu + 1).to(torch.float32)
    
    # Rearrange dimensions to [B, C, T]
    x_onehot = x_onehot.permute(0, 2, 1)
    
    return x_onehot

def mu_law_decode(x):
    """
    Decodes audio from μ-law encoded one-hot representation back to waveform.
    """
    mu = 255  # μ value for the encoding/decoding

    # Ensure μ is a tensor compatible with the input device and dtype
    mu_tensor = torch.tensor(mu, device=x.device, dtype=x.dtype)
    
    # Step 1: Convert one-hot representation to indices [0, 255]
    x_indices = torch.argmax(x, dim=1)  # [B, T]

    # Step 2: Map indices to range [-1, 1]
    x_mu = 2 * x_indices.float() / mu_tensor - 1  # [B, T]

    # Step 3: Apply μ-law expansion formula
    x = torch.sign(x_mu) * (1 / mu_tensor) * ((1 + mu_tensor)**torch.abs(x_mu) - 1)

    # Step 4: Add a channel dimension to match expected shape
    x = x.unsqueeze(1)  # [B, 1, T]

    return x

def one_hot_encode(x, num_classes=256):
    x = torch.clamp(x, min=-1.0, max=1.0)
    x = (x + 1) / 2 * (num_classes - 1)  # Rescale to [0, 255]
    x = x.long()  

    return F.one_hot(x, num_classes=num_classes).float()


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(Function):
    def __init__(self, target_size, dimension=0, value=0, pad_start=False):
        super(ConstantPad1d, self).__init__()
        self.target_size = target_size
        self.dimension = dimension
        self.value = value
        self.pad_start = pad_start

    def forward(self, input):
        self.num_pad = self.target_size - input.size(self.dimension)
        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = input.size()

        size = list(input.size())
        size[self.dimension] = self.target_size
        output = input.new(*tuple(size)).fill_(self.value)
        c_output = output

        # crop output
        if self.pad_start:
            c_output = c_output.narrow(self.dimension, self.num_pad, c_output.size(self.dimension) - self.num_pad)
        else:
            c_output = c_output.narrow(self.dimension, 0, c_output.size(self.dimension) - self.num_pad)

        c_output.copy_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(*self.input_size).zero_()
        cg_output = grad_output

        # crop grad_output
        if self.pad_start:
            cg_output = cg_output.narrow(self.dimension, self.num_pad, cg_output.size(self.dimension) - self.num_pad)
        else:
            cg_output = cg_output.narrow(self.dimension, 0, cg_output.size(self.dimension) - self.num_pad)

        grad_input.copy_(cg_output)
        return grad_input


def constant_pad_1d(input, target_size, dimension=0, value=0, pad_start=True):
    """
    Pads the input tensor to the target size along the specified dimension.
    Uses PyTorch's native pad function instead of custom autograd function.
    """
    input_size = input.size(dimension)
    diff = target_size - input_size
    if diff <= 0:
        return input

    # Create padding tuple
    pad_size = [0] * (2 * input.dim())  # initialize padding for all dimensions
    if pad_start:
        pad_size[2 * dimension] = diff  # pad at start of specified dimension
    else:
        pad_size[2 * dimension + 1] = diff  # pad at end of specified dimension
    
    # Reverse the padding tuple as F.pad expects last dim first
    pad_size = pad_size[::-1]
    
    # Apply padding
    return F.pad(input, pad_size, mode='constant', value=value)