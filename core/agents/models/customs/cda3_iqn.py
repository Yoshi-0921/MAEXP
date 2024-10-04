"""Source code for distributed attentional actor architecture (DA3) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import numpy as np
import torch
from numpy import typing as npt
from omegaconf import DictConfig
from torch import functional as F
from torch import nn

from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging

from ..hard_shrink_attention import HardShrinkBlock
from ..mlp import MLP
from ..vit import Block, PatchEmbed
from .da3_iqn import IQN_Head
from .iqn import CosineEmbeddingNetwork

logger = initialize_logging(__name__)


class CDA3_IQN(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        patched_size_x = input_shape[1] // config.model.patch_size
        patched_size_y = input_shape[2] // config.model.patch_size
        self.view_method = config.observation_area_mask
        self.map_SIZE_X, self.map_SIZE_Y = config.map.SIZE_X, config.map.SIZE_Y
        self.embedding_dim = config.model.embed_dim

        self.objects_channel = config.objects_channel
        self.agents_channel = config.agents_channel
        self.destination_channel = config.destination_channel
        in_chans = input_shape[0]
        if self.objects_channel:
            in_chans += config.type_objects
        if self.agents_channel:
            in_chans += config.num_agents
        if self.destination_channel:
            in_chans += 1

        self.patch_embed = PatchEmbed(
            patch_size=config.model.patch_size,
            in_chans=in_chans,
            embed_dim=self.embedding_dim,
        )

        self.pos_saliency_vector = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.neg_saliency_vector = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patched_size_x * patched_size_y + 2, self.embedding_dim)
        )

        block = HardShrinkBlock if config.model.attention == "hard" else Block
        self.blocks = nn.ModuleList(
            [
                block(
                    dim=self.embedding_dim,
                    num_heads=config.model.num_heads,
                    mlp_ratio=config.model.mlp_ratio,
                    **{"af_lambd": config.model.af_lambd}
                )
                for _ in range(config.model.block_loop)
            ]
        )

        self.norm = nn.LayerNorm(self.embedding_dim)
        self.pos_drl_head = IQN_Head(
            config=config, embedding_dim=self.embedding_dim, output_size=output_size
        )
        self.neg_drl_head = IQN_Head(
            config=config, embedding_dim=self.embedding_dim, output_size=output_size
        )

    def forward(self, state, external_taus: torch.Tensor = None):
        pos_saliency_vector, neg_saliency_vector = self.get_saliency_vector(state=state)
        # pos_q_values = F.relu(self.pos_drl_head(
        #     state_embeddings=pos_saliency_vector, external_taus=external_taus
        # ))
        # neg_q_values = F.relu(self.neg_drl_head(
        #     state_embeddings=neg_saliency_vector, external_taus=external_taus
        # ))
        pos_q_values = self.pos_drl_head(
            state_embeddings=pos_saliency_vector, external_taus=external_taus
        )
        neg_q_values = self.neg_drl_head(
            state_embeddings=neg_saliency_vector, external_taus=external_taus
        )
        q_values = pos_q_values - neg_q_values

        return q_values

    def forward_attn(self, state, external_taus: torch.Tensor = None):
        pos_saliency_vector, neg_saliency_vector, attns = self.get_saliency_vector(
            state=state, output_attns=True
        )

        # pos_q_values = F.relu(self.pos_drl_head(
        #     state_embeddings=pos_saliency_vector, external_taus=external_taus
        # ))
        # neg_q_values = F.relu(self.neg_drl_head(
        #     state_embeddings=neg_saliency_vector, external_taus=external_taus
        # ))
        pos_q_values = self.pos_drl_head(
            state_embeddings=pos_saliency_vector, external_taus=external_taus
        )
        neg_q_values = self.neg_drl_head(
            state_embeddings=neg_saliency_vector, external_taus=external_taus
        )
        q_values = pos_q_values - neg_q_values

        return q_values, [attns]

    def get_saliency_vector(self, state, output_attns: bool = False):
        x = self.state_encoder(state)

        out = self.patch_embed(x)
        pos_saliency_vector = self.pos_saliency_vector.expand(out.shape[0], -1, -1)
        neg_saliency_vector = self.neg_saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((pos_saliency_vector, neg_saliency_vector, out), dim=1)
        out = out + self.pos_embed

        attns = list()
        for blk in self.blocks:
            out, attn = blk.forward_attn(out)
            attns.append(attn.detach())

        out = self.norm(out)
        pos_saliency_vector = out[:, 0]
        neg_saliency_vector = out[:, 1]

        if output_attns:
            return pos_saliency_vector, neg_saliency_vector, attns

        return pos_saliency_vector, neg_saliency_vector

    def get_quantiles(
        self,
        state=None,
        state_embeddings: torch.Tensor = None,
        taus: torch.Tensor = None,
    ):
        saliency_vector = (
            state_embeddings
            if state_embeddings is not None
            else self.get_saliency_vector(state)
        )
        if state_embeddings is not None:
            pos_quantiles = self.pos_drl_head.get_quantiles(state_embeddings=state_embeddings, taus=taus)
            neg_quantiles = self.pos_drl_head.get_quantiles(state_embeddings=state_embeddings, taus=taus)
        else:
            pos_saliency_vector, neg_saliency_vector = self.get_saliency_vector(state)
            pos_quantiles = self.pos_drl_head.get_quantiles(state_embeddings=pos_saliency_vector, taus=taus)
            neg_quantiles = self.pos_drl_head.get_quantiles(state_embeddings=neg_saliency_vector, taus=taus)

        quantiles = pos_quantiles - neg_quantiles

        return quantiles

    def state_encoder(self, state):
        if self.view_method == "relative":
            relative_x = ObservationHandler.decode_relative_state(
                state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
            )
            if self.objects_channel:
                relative_x = torch.cat((relative_x, state["objects"]), dim=1)
            if self.agents_channel:
                agents_channel = ObservationHandler.decode_agents_channel(
                    state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
                )
                relative_x = torch.cat((relative_x, agents_channel), dim=1)
            if self.destination_channel:
                relative_x = torch.cat((relative_x, state["destination"].unsqueeze(1)), dim=1)
            return relative_x

        return state[self.view_method]
