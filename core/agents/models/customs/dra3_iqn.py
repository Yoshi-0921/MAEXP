"""Source code for distributed recurrent attentional actor architecture (DRA3) model.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import torch
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from .da3_iqn import DA3_IQN

logger = initialize_logging(__name__)


class DRA3_IQN(DA3_IQN):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__(config=config, input_shape=input_shape, output_size=output_size)
        self.saliency_vector = nn.Parameter(torch.zeros(1, self.embedding_dim))
        self.recurrent_module = nn.GRUCell(input_size=self.embedding_dim, hidden_size=self.embedding_dim)

    def forward(self, state, external_taus: torch.Tensor = None, hidden_vector: torch.Tensor = None):
        saliency_vector, hidden_vector = self.get_saliency_vector(state=state, hidden_vector=hidden_vector)
        q_values = self.drl_head(
            state_embeddings=saliency_vector, external_taus=external_taus
        )

        return q_values, hidden_vector

    def forward_attn(self, state, external_taus: torch.Tensor = None, hidden_vector: torch.Tensor = None):
        saliency_vector, attns, hidden_vector = self.get_saliency_vector(
            state=state, output_attns=True, hidden_vector=hidden_vector
        )

        q_values = self.drl_head(
            state_embeddings=saliency_vector, external_taus=external_taus
        )

        return q_values, [attns], hidden_vector

    def get_saliency_vector(self, state, output_attns: bool = False, hidden_vector: torch.Tensor = None):
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

        if output_attns:
            return saliency_vector, attns, hidden_vector

        return saliency_vector, hidden_vector

    def get_quantiles(
        self,
        state=None,
        state_embeddings: torch.Tensor = None,
        taus: torch.Tensor = None,
    ):
        saliency_vector, _ = (
            state_embeddings
            if state_embeddings is not None
            else self.get_saliency_vector(state)
        )

        return self.drl_head.get_quantiles(state_embeddings=saliency_vector, taus=taus)
