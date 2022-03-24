"""Source code for multi-agent transfromer (DA6) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from core.handlers.observations.observation_handler import ObservationHandler
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
        self.destination_channel = config.destination_channel
        relative_patched_size_x = input_shape[1] // config.model.relative_patch_size
        relative_patched_size_y = input_shape[2] // config.model.relative_patch_size
        local_patched_size_x = config.visible_range // config.model.local_patch_size
        local_patched_size_y = config.visible_range // config.model.local_patch_size
        self.map_SIZE_X, self.map_SIZE_Y = config.map.SIZE_X, config.map.SIZE_Y

        self.relative_patch_embed = PatchEmbed(
            patch_size=config.model.relative_patch_size,
            in_chans=2 if config.destination_channel else 1,
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
        self.drl_head = nn.Linear(config.model.embed_dim, output_size)

    def forward(self, state):
        local_x, relative_x = self.state_encoder(state)

        out = self.relative_patch_embed(relative_x)
        saliency_vector = self.saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.relative_pos_embed

        for blk in self.relative_blocks:
            out = blk(out)

        out = self.norm(out)
        saliency_vector = out[:, 0]
        saliency_vector = self.fc1(saliency_vector)

        residual_saliency_vector = saliency_vector.clone()

        out = self.local_patch_embed(local_x)
        saliency_vector = saliency_vector.unsqueeze(1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.local_pos_embed

        for blk in self.local_blocks:
            out = blk(out)

        out = self.norm(out)

        saliency_vector = out[:, 0]

        saliency_vector += residual_saliency_vector

        out = self.drl_head(saliency_vector)

        return out

    def forward_attn(self, state):
        local_x, relative_x = self.state_encoder(state)

        out = self.relative_patch_embed(relative_x)
        saliency_vector = self.saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.relative_pos_embed

        relative_attns: List[npt.NDArray[np.float32]] = list()
        for blk in self.relative_blocks:
            out, attn = blk.forward_attn(out)
            relative_attns.append(attn.detach())

        out = self.norm(out)
        saliency_vector = out[:, 0]
        saliency_vector = self.fc1(saliency_vector)

        residual_saliency_vector = saliency_vector.clone()

        out = self.local_patch_embed(local_x)
        saliency_vector = saliency_vector.unsqueeze(1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.local_pos_embed

        local_attns: List[npt.NDArray[np.float32]] = list()
        for blk in self.local_blocks:
            out, attn = blk.forward_attn(out)
            local_attns.append(attn.detach())

        out = self.norm(out)

        saliency_vector = out[:, 0]

        saliency_vector += residual_saliency_vector

        out = self.drl_head(saliency_vector)

        return out, [local_attns, relative_attns]

    def state_encoder(self, state):
        local_x = state["local"]
        # x.shape: [1, 4, 25, 25]
        relative_x = ObservationHandler.decode_relative_state(
            state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
        )
        relative_x = relative_x[:, -1:, ...]  # [1, 1, 25, 25]が欲しい
        relative_x += 1

        if self.destination_channel:
            destination_channel = state["destination_channel"][:, -1:, ...]
            relative_x = torch.cat((relative_x, destination_channel), dim=1)

        return local_x, relative_x
