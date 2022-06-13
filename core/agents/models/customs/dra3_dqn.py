"""Source code for distributed attentional actor architecture (DA3) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import torch
from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn
from .da3 import DA3
from ..hard_shrink_attention import HardShrinkBlock
from ..vit import Block, PatchEmbed

logger = initialize_logging(__name__)


class DRA3_DQN(DA3):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__(config=config, input_shape=input_shape, output_size=output_size)
        self.saliency_vector = nn.Parameter(torch.zeros(1, 1, config.model.embed_dim))
        self.recurrent_module = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim)

    def forward(self, state, hidden_vector: torch.Tensor = None):
        out, _, hidden_vector = self.forward_attn(state=state, hidden_vector=hidden_vector)

        return out, hidden_vector

    def forward_attn(self, state, hidden_vector: torch.Tensor = None):
        x = self.state_encoder(state)
        out = self.patch_embed(x)

        if hidden_vector is None:
            hidden_vector = torch.rand(size=(out.shape[0], self.embedding_dim), device=out.device)

        saliency_vector = self.saliency_vector.expand(out.shape[0], -1)
        hidden_vector = self.recurrent_module(saliency_vector, hidden_vector)
        saliency_vector = hidden_vector.unsqueeze(1).expand(out.shape[0], -1, -1)

        out = torch.cat((saliency_vector, out), dim=1)
        out = out + self.pos_embed

        attns = list()
        for blk in self.blocks:
            out, attn = blk.forward_attn(out)
            attns.append(attn.detach())

        out = self.norm(out)
        saliency_vector = out[:, 0]

        out = self.head(saliency_vector)

        return out, [attns], hidden_vector
