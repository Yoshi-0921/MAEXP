# -*- coding: utf-8 -*-

"""Source code for multi-agent transfromer (DA6) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import torch
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from ..hard_shrink_attention import HardShrinkBlock
from ..vit import Block, PatchEmbed

logger = initialize_logging(__name__)


class DA6(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.visible_range = config.visible_range
        relative_patched_size_x = input_shape[1] // config.model.relative_patch_size
        relative_patched_size_y = input_shape[2] // config.model.relative_patch_size
        local_patched_size_x = config.visible_range // config.model.local_patch_size
        local_patched_size_y = config.visible_range // config.model.local_patch_size

        self.relative_patch_embed = PatchEmbed(
            patch_size=config.model.relative_patch_size,
            in_chans=1,
            embed_dim=config.model.embed_dim,
        )
        self.local_patch_embed = PatchEmbed(
            patch_size=config.model.local_patch_size,
            in_chans=input_shape[0],
            embed_dim=config.model.embed_dim,
        )

        self.saliency_vector = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
        self.relative_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                relative_patched_size_x * relative_patched_size_y + 1,
                config.model.embed_dim,
            )
        )
        self.local_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                local_patched_size_x * local_patched_size_y + 1,
                config.model.embed_dim,
            )
        )

        block = HardShrinkBlock if config.model.attention == "hard" else Block
        self.relative_blocks = nn.ModuleList(
            [
                block(
                    dim=config.model.embed_dim,
                    num_heads=config.model.num_heads,
                    mlp_ratio=config.model.mlp_ratio,
                    **{"af_lambd": config.model.af_lambd}
                )
                for _ in range(config.model.block_loop)
            ]
        )
        self.local_blocks = nn.ModuleList(
            [
                block(
                    dim=config.model.embed_dim,
                    num_heads=config.model.num_heads,
                    mlp_ratio=config.model.mlp_ratio,
                    **{"af_lambd": config.model.af_lambd}
                )
                for _ in range(config.model.block_loop)
            ]
        )

        self.norm = nn.LayerNorm(config.model.embed_dim)
        self.fc1 = nn.Linear(config.model.embed_dim, config.model.embed_dim)
        self.fc2 = nn.Linear(config.model.embed_dim, output_size)

    def forward(self, relative_x, local_x):
        relative_x, local_x = self.state_encoder(relative_x, local_x)

        out = self.relative_patch_embed(relative_x)
        saliency_vector = self.saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.relative_pos_embed

        for blk in self.relative_blocks:
            out = blk(out)

        out = self.norm(out)
        saliency_vector = out[:, 0]
        saliency_vector = self.fc1(saliency_vector)

        out = self.local_patch_embed(local_x)
        saliency_vector = saliency_vector.unsqueeze(1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.local_pos_embed

        for blk in self.local_blocks:
            out = blk(out)

        out = self.norm(out)

        saliency_vector = out[:, 0]

        out = self.fc2(saliency_vector)

        return out

    def forward_attn(self, relative_x, local_x):
        relative_x, local_x = self.state_encoder(relative_x, local_x)

        out = self.relative_patch_embed(relative_x)
        saliency_vector = self.saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.relative_pos_embed

        for blk in self.relative_blocks:
            out = blk(out)

        out = self.norm(out)
        saliency_vector = out[:, 0]
        saliency_vector = self.fc1(saliency_vector)

        out = self.local_patch_embed(local_x)
        saliency_vector = saliency_vector.unsqueeze(1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.local_pos_embed

        attns = list()
        for blk in self.local_blocks:
            out, attn = blk.forward_attn(out)
            attns.append(attn.detach())

        out = self.norm(out)

        saliency_vector = out[:, 0]

        out = self.fc2(saliency_vector)

        return out, attns

    def state_encoder(self, relative_x, local_x):
        # x.shape: [1, 4, 25, 25]
        relative_x = relative_x[:, -1, ...].unsqueeze(1)  # [1, 1, 25, 25]が欲しい
        relative_x += 1

        return relative_x, local_x
