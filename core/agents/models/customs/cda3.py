"""Source code for contextual distributed attentional actor architecture (DA3) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging

from ..hard_shrink_attention import HardShrinkBlock
from ..vit import Block, PatchEmbed

logger = initialize_logging(__name__)


class CDA3(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        patched_size_x = input_shape[1] // config.model.patch_size
        patched_size_y = input_shape[2] // config.model.patch_size
        self.view_method = config.observation_area_mask
        self.map_SIZE_X, self.map_SIZE_Y = config.map.SIZE_X, config.map.SIZE_Y
        self.embedding_dim = config.model.embed_dim

        self.objects_channel = config.objects_channel
        in_chans = input_shape[0]
        if self.objects_channel:
            in_chans += config.type_objects
        self.patch_embed = PatchEmbed(
            patch_size=config.model.patch_size,
            in_chans=in_chans,
            embed_dim=config.model.embed_dim,
        )

        self.pos_saliency_vector = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
        self.neg_saliency_vector = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patched_size_x * patched_size_y + 2, config.model.embed_dim)
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
        self.pos_head = nn.Linear(config.model.embed_dim, output_size)
        self.neg_head = nn.Linear(config.model.embed_dim, output_size)

    def forward(self, state):
        x = self.state_encoder(state)

        out = self.patch_embed(x)
        pos_saliency_vector = self.pos_saliency_vector.expand(out.shape[0], -1, -1)
        neg_saliency_vector = self.neg_saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((pos_saliency_vector, neg_saliency_vector, out), dim=1)
        out = out + self.pos_embed

        for blk in self.blocks:
            out = blk(out)

        out = self.norm(out)
        pos_out = out[:, 0]
        neg_out = out[:, 1]

        # pos_out = F.relu(self.pos_head(pos_out))
        # neg_out = F.relu(self.neg_head(neg_out))
        pos_out = self.pos_head(pos_out)
        neg_out = self.neg_head(neg_out)
        out = pos_out - neg_out

        return out

    def forward_attn(self, state):
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
        pos_out = out[:, 0]
        neg_out = out[:, 1]

        # pos_out = F.relu(self.pos_head(pos_out))
        # neg_out = F.relu(self.neg_head(neg_out))
        pos_out = self.pos_head(pos_out)
        neg_out = self.neg_head(neg_out)
        out = pos_out - neg_out

        return out, [attns]

    def state_encoder(self, state):
        if self.view_method == "relative":
            relative_x = ObservationHandler.decode_relative_state(
                state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
            )
            if self.objects_channel:
                relative_x = torch.cat((relative_x, state["objects"]), dim=1)

            return relative_x

        return state[self.view_method]