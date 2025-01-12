"""
Modules that combine continuous and discrete latent space components
"""
from typing import Union
import torch
import torch.nn as nn
from einops import rearrange

from src.models.attention import EfficientCrossAttention1D

class MLPCombiner(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv1d(2*self.channels, 2*self.channels, kernel_size=3, dilation=1, padding='same'),
            nn.BatchNorm1d(2*self.channels),
            nn.ELU(),
            nn.Conv1d(2*self.channels, 2*self.channels, kernel_size=3, dilation=3, padding='same'),
            nn.BatchNorm1d(2*self.channels),
            nn.ELU(),
            nn.Conv1d(2*self.channels, self.channels, kernel_size=3, dilation=5, padding='same'),
            nn.BatchNorm1d(2*self.channels),
            nn.ELU()
        )
        
    def forward(self, z1, z2):
        # assume z1, z2 have shape [B, emb_dim, L]
        z = torch.cat((z1, z2), dim=1)
        return self.net(z)
    
    
class CACombiner(nn.Module):
    def __init__(self, in_ch, key_ch, value_ch, n_heads=8):
        super().__init__()
        self.ca = EfficientCrossAttention1D(in_channels=in_ch, key_channels=key_ch, value_channels=value_ch, head_count=n_heads)
    
    def forward(self, z1, z2):
        # assume z1 = z_recon, z2 = z_vq shape [B, latent_emb_ch, L]
        # rearrange if we used nn.Linear instead
        # z1 = rearrange(z1, 'b c l -> b l c')
        # z2 = rearrange(z2, 'b c l -> b l c')
        z = self.ca(z1, z2)
        # z = rearrange(z, 'b l c -> b c l')
        return z