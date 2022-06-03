"""Source code for fully parameterized quantile function (FQF) layers class.

Original code: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
"""

from typing import List

import torch
import torch.nn.functional as F
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from .iqn import IQN

logger = initialize_logging(__name__)


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class FractionProposalNetwork(nn.Module):
    def __init__(self, num_quantiles, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, num_quantiles)
        ).apply(lambda x: initialize_weights_xavier(x, gain=0.01))

        self.num_quantiles = num_quantiles
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings):
        batch_size = state_embeddings.shape[0]

        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.num_quantiles)

        tau_0 = torch.zeros(
            (batch_size, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        taus_1_N = torch.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.num_quantiles + 1)

        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        assert tau_hats.shape == (batch_size, self.num_quantiles)

        # Calculate entropies of value distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)

        return taus, tau_hats, entropies


class FQF(IQN):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int, target: bool):
        super().__init__(config=config, input_shape=input_shape, output_size=output_size)
        if not target:
            self.fraction_net = FractionProposalNetwork(num_quantiles=self.num_quantiles, embedding_dim=self.embedding_dim)

    def forward(self, state, external_taus: torch.Tensor = None, external_tau_hats: torch.Tensor = None):
        state_embeddings = self.get_state_embeddings(state=state)
        taus, tau_hats, _ = self.get_taus(state_embeddings=state_embeddings)
        taus = external_taus if external_taus is not None else taus
        tau_hats = external_tau_hats if external_tau_hats is not None else tau_hats
        quantile_hats = self.get_quantiles(state_embeddings=state_embeddings, taus=tau_hats)
        q_values = self.get_q_values(quantiles=quantile_hats, taus=taus)

        return q_values

    def get_taus(self, state_embeddings):
        return self.fraction_net(state_embeddings)

    def get_q_values(self, quantiles: torch.Tensor, taus: torch.Tensor):
        q_values = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantiles).sum(dim=1)

        return q_values
