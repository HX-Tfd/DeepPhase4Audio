import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from inspect import isfunction


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 use_fp16=False, emb_channels=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = self._default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.affine = nn.Linear(emb_channels, inner_dim * 2) if emb_channels is not None else None

        if use_fp16:
            self.to_q = self.to_q.half()
            self.to_k = self.to_k.half()
            self.to_v = self.to_v.half()
            self.to_out = self.to_out.half()
    

    @staticmethod
    def _default(val, d):
        def exists(val):
            return val is not None
        if exists(val):
            return val
        return d() if isfunction(d) else d

    def forward(self, x, context=None, mask=None, t_emb=None):
        h = self.heads

        q = self.to_q(x)
        context = self._default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)

        return out
   
   
class EfficientCrossAttention1D(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels, use_fp16=False):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv1d(in_channels, key_channels, 1)
        self.queries = nn.Conv1d(in_channels, key_channels, 1)
        self.values = nn.Conv1d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv1d(self.value_channels, in_channels, 1)
        
        if use_fp16:
            self.keys = self.keys.half()
            self.queries = self.queries.half()
            self.values = self.values.half()
            self.reprojection = self.reprojection.half()


    def forward(self, input_, context_):
        # Assuming input_ and context_ are of shape [n, c, l]
        b, _, l = input_.size()  # (batch size, channels, sequence length)

        # Compute keys, queries, and values (shape [n, channels, l])
        keys = self.keys(context_)      # [n, key_channels, l]
        queries = self.queries(input_)  # [n, key_channels, l]
        values = self.values(context_)  # [n, value_channels, l]

        # Split the key and value channels for attention heads
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            # Q, K, V for each head
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)  # [n, head_key_channels, l]
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)  # [n, head_key_channels, l]
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]  # [n, head_value_channels, l]

            # Cross-attention: key @ value (scaled dot-product)
            context = key @ value.transpose(1, 2)  # [n, head_key_channels, head_value_channels]

            # Attend to the context using query
            attended_value = (context.transpose(1, 2) @ query).reshape(b, head_value_channels, l)
            attended_values.append(attended_value)

        # Concatenate attended values across heads
        aggregated_values = torch.cat(attended_values, dim=1)     # [n, l, heads]
        reprojected_value = self.reprojection(aggregated_values)  # [n, in_channels, l]
        attention = reprojected_value + input_                    # [n, in_channels, l]

        return attention
 
    
class EfficientCrossAttention2D(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        # Convolutions for keys, queries, and values
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_, context_):

        n, _, h, w = input_.size()

        # Compute keys, queries, and values
        keys = self.keys(context_).reshape((n, self.key_channels, h * w))  # [n, key_channels, h*w]
        queries = self.queries(input_).reshape((n, self.key_channels, h * w))  # [n, key_channels, h*w]
        values = self.values(context_).reshape((n, self.value_channels, h * w))  # [n, value_channels, h*w]

        # Split the key and value channels for attention heads
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            # Select K, Q, V for each head
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)  # [n, head_key_channels, h*w]
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)  # [n, head_key_channels, h*w]
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]  # [n, head_value_channels, h*w]

            # xattn: key @ value (scaled dot-product)
            context = key @ value.transpose(1, 2)  # [n, head_key_channels, head_value_channels]

            # Attend to the context using query
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        # Concatenate attended values across heads
        aggregated_values = torch.cat(attended_values, dim=1)  # [n, head_value_channels * head_count, h, w]

        # Reprojection to the original input channels
        reprojected_value = self.reprojection(aggregated_values)

        # Add the original input to the attended value
        attention = reprojected_value + input_  # [n, in_channels, h, w]

        return attention


if __name__ == '__main__':
    # test 2D
    # B, C, H, W = 16, 128, 512, 512
    # input_tensor = torch.rand(B, C, H, W)    # Query   [B, C, H, W]
    # context_tensor = torch.rand(B, C, H, W)  # Context [B, C, H, W]
    # attention_layer = EfficientCrossAttention2D(in_channels=C, key_channels=64, head_count=8, value_channels=64)
    # output = attention_layer(input_tensor, context_tensor)
    
    # print(output.shape)  # [B, C, H, W]
    
    # test 1D
    B, C, L = 16, 128, 32000
    input_tensor = torch.rand(B, C, L)    # Query  [B, C, L]
    context_tensor = torch.rand(B, C, L)  # Context [B, C, L]

    # Initialize and run Efficient Cross-Attention
    attention_layer = EfficientCrossAttention1D(in_channels=C, key_channels=64, head_count=8, value_channels=64)
    output = attention_layer(input_tensor, context_tensor)

    print(output.shape)  # [B, C, L]