"""Source code for distributed attentional actor architecture after another attention (DA6) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import numpy as np
import torch
from numpy import typing as npt
from omegaconf import DictConfig
from torch import nn

from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging

from ..hard_shrink_attention import HardShrinkBlock
from ..vit import Block, PatchEmbed
from .da3_iqn import DA3_IQN, IQN_Head

logger = initialize_logging(__name__)


class DA6_IQN(DA3_IQN):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__(
            config=config, input_shape=input_shape, output_size=output_size
        )
        relative_patched_size_x = (
            input_shape[1] // 5
        )  # config.model.relative_patch_size
        relative_patched_size_y = (
            input_shape[2] // 5
        )  # config.model.relative_patch_size
        local_patched_size_x = (
            config.visible_range // 1
        )  # config.model.local_patch_size
        local_patched_size_y = (
            config.visible_range // 1
        )  # config.model.local_patch_size

        self.objects_channel = config.objects_channel
        self.agents_channel = config.agents_channel
        in_chans = 1
        if self.objects_channel:
            in_chans += config.type_objects
        if self.agents_channel:
            in_chans += config.num_agents
        self.relative_patch_embed = PatchEmbed(
            patch_size=5,  # config.model.relative_patch_size,
            in_chans=in_chans,
            embed_dim=config.model.embed_dim,
        )
        self.local_patch_embed = PatchEmbed(
            patch_size=1,  # config.model.local_patch_size,
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
        self.drl_head = IQN_Head(
            config=config, embedding_dim=self.embedding_dim, output_size=output_size
        )

    def forward_attn(self, state, external_taus: torch.Tensor = None):
        saliency_vector, local_attns, relative_attns = self.get_saliency_vector(
            state=state, output_attns=True
        )

        q_values = self.drl_head(
            state_embeddings=saliency_vector, external_taus=external_taus
        )

        return q_values, [local_attns, relative_attns]

    def get_saliency_vector(self, state, output_attns: bool = False):
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

        if output_attns:
            return saliency_vector, local_attns, relative_attns

        return saliency_vector

    def state_encoder(self, state):
        local_x = state["local"]
        # x.shape: [1, 4, 25, 25]
        relative_x = ObservationHandler.decode_relative_state(
            state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
        )
        relative_x = relative_x[:, -1:, ...]  # [1, 1, 25, 25]が欲しい
        relative_x += 1

        if self.objects_channel:
            relative_x = torch.cat((relative_x, state["objects"]), dim=1)

        if self.agents_channel:
            agents_channel = ObservationHandler.decode_agents_channel(
            state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
        )
            relative_x = torch.cat((relative_x, agents_channel), dim=1)

        return local_x, relative_x
