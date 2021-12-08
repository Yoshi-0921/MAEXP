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
from .iqn import CosineEmbeddingNetwork
from ..mlp import MLP

logger = initialize_logging(__name__)


class IQN_Head(nn.Module):
    def __init__(self, config: DictConfig, embedding_dim: int, output_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_actions = output_size
        self.num_quantiles: int = config.model.num_quantiles
        self.num_cosines: int = config.model.num_cosines
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=self.num_cosines, embedding_dim=embedding_dim
        )
        self.fc_V = MLP(
            config=config, input_size=self.num_quantiles * embedding_dim, output_size=self.num_quantiles
        )
        self.fc_A = MLP(
            config=config,
            input_size=self.num_quantiles * embedding_dim,
            output_size=output_size * self.num_quantiles
        )

    def forward(self, state_embeddings, taus):
        batch_size = state_embeddings.shape[0]

        tau_embeddings = self.cosine_net(taus)
        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(batch_size, 1, self.embedding_dim)
        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size, self.num_quantiles * self.embedding_dim
        )

        V = self.fc_V(embeddings)
        A = self.fc_A(embeddings)

        V = V.view(batch_size, 1, self.num_quantiles)
        A = A.view(batch_size, self.num_actions, self.num_quantiles)

        average_A = A.mean(1, keepdim=True)
        quantiles = V.expand_as(A) + (A - average_A.expand_as(A))

        return quantiles


class DA3_IQN(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        patched_size_x = input_shape[1] // config.model.patch_size
        patched_size_y = input_shape[2] // config.model.patch_size
        self.view_method = config.observation_area_mask
        self.embedding_dim = config.model.embed_dim

        self.patch_embed = PatchEmbed(
            patch_size=config.model.patch_size,
            in_chans=input_shape[0],
            embed_dim=self.embedding_dim,
        )

        self.saliency_vector = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patched_size_x * patched_size_y + 1, self.embedding_dim)
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
        self.drl_head = IQN_Head(config=config, embedding_dim=self.embedding_dim, output_size=output_size)

    def forward(self, state, taus):
        x = self.state_encoder(state)

        out = self.patch_embed(x)
        saliency_vector = self.saliency_vector.expand(out.shape[0], -1, -1)
        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.pos_embed

        for blk in self.blocks:
            out = blk(out)

        out = self.norm(out)
        saliency_vector = out[:, 0]

        out = self.drl_head(saliency_vector, taus)

        return out

    def forward_attn(self, state, taus):
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

        out = self.drl_head(saliency_vector, taus)

        return out, [attns]

    def state_encoder(self, state):

        return state[self.view_method]
