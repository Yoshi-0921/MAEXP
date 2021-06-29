# -*- coding: utf-8 -*-

"""Source code for multi-agent splited transfromer (MAST) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import torch
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from ..vit import Block

logger = initialize_logging(__name__)


class SplitPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=1, in_chans=3, embed_dim=128):
        super().__init__()
        self.in_channels = [1 for _ in range(in_chans)]
        self.projs = nn.ModuleList([nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size) for _ in range(in_chans)])

    def forward(self, x):
        xs = torch.split(x, self.in_channels, 1)
        x = torch.stack([proj(x).flatten(2).transpose(1, 2) for proj, x in zip(self.projs, xs)], 1)

        return x


class SplitBlock(nn.Module):
    def __init__(self, channels, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.in_channels = [1 for _ in range(channels)]
        self.channel_blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer, **kwargs) for _ in range(channels)
        ])

    def forward(self, x):
        xs = torch.split(x, self.in_channels, 1)
        x = torch.stack([blk(x.squeeze(1)) for blk, x in zip(self.channel_blocks, xs)], 1)

        return x

    def forward_attn(self, x):
        outs, attns = [], []
        xs = torch.split(x, self.in_channels, 1)
        for blk, x in zip(self.channel_blocks, xs):
            out, attn = blk.forward_attn(x.squeeze(1))
            outs.append(out)
            attns.append(attn)

        return torch.stack(outs, 1), torch.stack(attns, 1)


class MAST(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.channels = input_shape[0]
        patched_size_x = input_shape[1] // config.model.patch_size
        patched_size_y = input_shape[2] // config.model.patch_size

        self.patch_embed = SplitPatchEmbed(
            patch_size=config.model.patch_size,
            in_chans=input_shape[0],
            embed_dim=config.model.embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
        self.final_token = nn.Parameter(torch.zeros(1, config.model.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, patched_size_x * patched_size_y + 1, config.model.embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                SplitBlock(
                    channels=self.channels,
                    dim=config.model.embed_dim,
                    num_heads=config.model.num_heads,
                    mlp_ratio=config.model.mlp_ratio
                )
                for _ in range(config.model.block_loop)
            ]
        )
        self.final_blocks = nn.ModuleList(
            [
                Block(
                    dim=config.model.embed_dim,
                    num_heads=config.model.num_heads,
                    mlp_ratio=config.model.mlp_ratio,
                    **{'af_lambd': config.model.af_lambd}
                )
                for _ in range(config.model.block_loop)
            ]
        )

        self.norm = nn.LayerNorm(config.model.embed_dim)
        self.fc1 = nn.Linear(config.model.embed_dim, output_size)

    def forward(self, x):
        out = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(out.shape[0], self.channels, -1, -1)
        out = torch.cat((cls_tokens, out), dim=2)
        out = out + self.pos_embed

        for blk in self.blocks:
            out = blk(out)

        out = self.norm(out)
        out = out[:, :, 0]

        final_tokens = self.final_token.expand(out.shape[0], -1, -1)
        out = torch.cat((final_tokens, out), dim=1)

        for blk in self.final_blocks:
            out = blk(out)

        out = self.norm(out)
        out = out[:, 0]

        out = self.fc1(out)

        return out

    def forward_attn(self, x):
        out = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(out.shape[0], self.channels, -1, -1)
        out = torch.cat((cls_tokens, out), dim=2)
        out = out + self.pos_embed

        attns, merged_attns = list(), list()
        for blk in self.blocks:
            out, attn = blk.forward_attn(out)
            attns.append(attn.detach())

        out = self.norm(out)
        out = out[:, :, 0]

        final_tokens = self.final_token.expand(out.shape[0], -1, -1)
        out = torch.cat((final_tokens, out), dim=1)

        for blk in self.final_blocks:
            out, attn = blk.forward_attn(out)
            merged_attns.append(attn.detach())

        out = self.norm(out)
        out = out[:, 0]

        out = self.fc1(out)

        return out, [attns, merged_attns]
