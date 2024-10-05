"""Source code for vision transformer with attention on attention.

"""
import torch.nn as nn
from .vit import Mlp
import torch

class AoABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.aoa_layer = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.GLU())

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        # Attention section
        B, N, C = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Main section
        out = self.attn(x, q, k, v)
        out = self.aoa_layer(torch.cat([out, q], -1))
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_attn(self, x):
        # Attention section
        B, N, C = x.shape
        qkv = (
            self.qkv(self.norm1(x))
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Main section
        out, attn = self.attn.forward_attn(x, q, k, v)

        q = q.transpose(1, 2).reshape(B, N, C)
        out = self.aoa_layer(torch.cat([out, q], -1))

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x, attn


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q, k, v):
        B, N, C = x.shape

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attn(self, x, q, k, v):
        B, N, C = x.shape

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
