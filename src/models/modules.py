import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


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
    

def activation(name: str, **kwargs) -> nn.Module:
    return {
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU,
        'snake': Snake
    }[name](**kwargs)


def positional_encoding(pos: torch.Tensor, d_model=1):  # pos is a tensor of shape (B, L)
    """
    Adds 1D sinusoidal positional encoding to the input positions.
    Args:
    pos (torch.Tensor): Input tensor of shape (B, L), representing positions.
    d_model (int): Dimensionality of the positional encoding.
    
    Returns:
    torch.Tensor: Positional encoding of shape (B, L, d_model).
    """
    B, L = pos.shape
    position = torch.arange(L, device=pos.device).unsqueeze(1).float()  # (L, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=pos.device).float() * 
                         -(torch.log(torch.tensor(10000.0)) / d_model))  # (d_model / 2,)
    pe = torch.zeros((L, d_model), device=pos.device)  # (L, d_model)

    # Apply sine and cosine functions to even and odd indices
    pe[:, 0::2] = torch.sin(position * div_term)  
    if d_model > 1: 
        pe[:, 1::2] = torch.cos(position * div_term)  

    return pe.unsqueeze(0).expand(B, -1, -1).squeeze()  # (B, L, d_model)

    
def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Snake(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return x + (1 / self.alpha) * torch.sin(self.alpha * x) ** 2
    

@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)