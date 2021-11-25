"""Source code for distributed attentional actor architecture (DA3) model.

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


class DA3(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        patched_size_x = input_shape[1] // config.model.patch_size
        patched_size_y = input_shape[2] // config.model.patch_size

        self.patch_embed = PatchEmbed(
            patch_size=config.model.patch_size,
            in_chans=input_shape[0],
            embed_dim=config.model.embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patched_size_x * patched_size_y + 1, config.model.embed_dim)
        )

        block = HardShrinkBlock if config.model.attention == "hard" else Block
        self.blocks = nn.ModuleList(
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
        self.head = nn.Linear(config.model.embed_dim, output_size)

    def forward(self, x):
        out = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + self.pos_embed

        for blk in self.blocks:
            out = blk(out)

        out = self.norm(out)
        out = out[:, 0]

        out = self.head(out)

        return out

    def forward_attn(self, x):
        out = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + self.pos_embed

        attns = list()
        for blk in self.blocks:
            out, attn = blk.forward_attn(out)
            attns.append(attn.detach())

        out = self.norm(out)
        out = out[:, 0]

        out = self.head(out)

        return out, attns
