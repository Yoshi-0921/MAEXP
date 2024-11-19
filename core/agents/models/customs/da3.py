"""Source code for distributed attentional actor architecture (DA3) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging

from ..hard_shrink_attention import HardShrinkBlock
from ..vit import Block, PatchEmbed
from ..convolution import Conv
from ..mlp import MLP

logger = initialize_logging(__name__)


class DA3(nn.Module):
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

        self.saliency_vector = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
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
        self.drl_head = nn.Linear(config.model.embed_dim, output_size)

    def forward(self, state):
        saliency_vector = self.get_saliency_vector(state=state)
        q_values = self.drl_head(saliency_vector)

        return q_values

    def forward_attn(self, state):
        saliency_vector, attns = self.get_saliency_vector(
            state=state, output_attns=True
        )

        q_values = self.drl_head(saliency_vector)

        return q_values, [attns]

    def get_saliency_vector(self, state, output_attns: bool = False):
        x = self.state_encoder(state)

        out = self.patch_embed(x)
        saliency_vector = self.saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.pos_embed

        attns = list()
        for blk in self.blocks:
            out, attn = blk.forward_attn(out)
            attns.append(attn.detach())

        out = self.norm(out)
        saliency_vector = out[:, 0]

        if output_attns:
            return saliency_vector, attns

        return saliency_vector


    def state_encoder(self, state):
        if self.view_method == "relative":
            relative_x = ObservationHandler.decode_relative_state(
                state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
            )
            if self.objects_channel:
                relative_x = torch.cat((relative_x, state["objects"]), dim=1)

            return relative_x

        return state[self.view_method]


class MergedDA3(DA3):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__(
            config=config, input_shape=input_shape, output_size=output_size
        )
        self.visible_range = config.visible_range

        local_patched_size_x = (
            config.visible_range // 1
        )  # config.model.local_patch_size
        local_patched_size_y = (
            config.visible_range // 1
        )  # config.model.local_patch_size

        self.local_patch_embed = PatchEmbed(
            patch_size=1,  # config.model.local_patch_size,
            in_chans=input_shape[0],
            embed_dim=config.model.embed_dim,
        )

        self.local_saliency_vector = nn.Parameter(
            torch.zeros(1, 1, config.model.embed_dim)
        )

        self.local_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                local_patched_size_x * local_patched_size_y + 1,
                config.model.embed_dim,
            )
        )

        block = HardShrinkBlock if config.model.attention == "hard" else Block
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

        self.relative_conv = Conv(
            config=config,
            input_channel=1,
            output_channel=config.model.output_channel,
        )
        relative_embedding_dim: int = MergedDA3.get_mlp_input_size(self.relative_conv, [1, *input_shape[1:]])
        self.relative_state_embedder = MLP(
            config=config, input_size=relative_embedding_dim, output_size=relative_embedding_dim
        )
        self.drl_head = nn.Linear(config.model.embed_dim+relative_embedding_dim, output_size)


    @staticmethod
    def get_mlp_input_size(conv, input_shape: List[int]) -> int:
        random_input = torch.randn(size=input_shape).unsqueeze(0)
        outputs = conv(random_input)

        return outputs.view(-1).shape[0]

    def forward_attn(self, state):
        saliency_vector, local_attns, relative_attns = self.get_saliency_vector(
            state=state, output_attns=True
        )

        q_values = self.drl_head(saliency_vector)

        return q_values, [local_attns, relative_attns]

    def get_saliency_vector(self, state, output_attns: bool = False):
        local_x, relative_x = self.state_encoder(state)
        relative_out = self.relative_conv(relative_x)
        relative_out = relative_out.view(relative_out.shape[0], -1)
        relative_saliency_vector = self.relative_state_embedder(relative_out)

        out = self.local_patch_embed(local_x)
        local_saliency_vector = self.local_saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((local_saliency_vector, out), dim=1)
        out = out + self.local_pos_embed

        local_attns = list()
        for blk in self.local_blocks:
            out, attn = blk.forward_attn(out)
            local_attns.append(attn.detach())

        out = self.norm(out)

        local_saliency_vector = out[:, 0]

        saliency_vector = torch.cat(
            (relative_saliency_vector, local_saliency_vector), dim=1
        )

        if output_attns:
            # return saliency_vector, local_attns, relative_attns
            return saliency_vector, local_attns, local_attns

        return saliency_vector

    def state_encoder(self, state):
        local_x = state["local"]
        # x.shape: [1, 4, 25, 25]
        relative_x = ObservationHandler.decode_relative_state(
            state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
        )
        relative_x = relative_x[:, -1:, ...]  # [1, 1, 25, 25]が欲しい
        relative_x += 1
        return local_x, relative_x
