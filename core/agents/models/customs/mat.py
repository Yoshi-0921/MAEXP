# -*- coding: utf-8 -*-

"""Source code for MAT.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from torch import nn
import torch
from ..vit import PatchEmbed, Block
from omegaconf import DictConfig
from typing import List


class MAT(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.patch_embed = PatchEmbed(
            patch_size=config.model.patch_size,
            in_chans=input_shape[0],
            embed_dim=config.model.embed_dim,
        )
        self.patched_size_x = input_shape[1] // config.model.patch_size
        self.patched_size_y = input_shape[2] // config.model.patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patched_size_x * self.patched_size_y + 1, config.model.embed_dim)
        )

        self.blocks = nn.ModuleList(
            [Block(dim=config.model.embed_dim, num_heads=4) for _ in range(1)]
        )

        self.norm = nn.LayerNorm(config.model.embed_dim)
        self.fc1 = nn.Linear(config.model.embed_dim, output_size)

    def forward(self, x):
        out = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + self.pos_embed

        attns = list()
        for blk in self.blocks:
            out, attn = blk(out)
            attns.append(attn)

        out = self.norm(out)
        out = out[:, 0]

        out = self.fc1(out)

        return out, attns
