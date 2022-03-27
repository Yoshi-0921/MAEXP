"""Source code for implicit quantile network (IQN) layers class.

Original code: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
"""

from typing import List

import torch
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from ..convolution import Conv
from ..mlp import MLP

logger = initialize_logging(__name__)


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = torch.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        return tau_embeddings


class IQN(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.view_method = config.observation_area_mask
        self.num_actions = output_size
        self.conv = Conv(
            config=config,
            input_channel=input_shape[0],
            output_channel=config.model.output_channel,
        )
        self.embedding_dim: int = self.get_mlp_input_size(input_shape)
        self.num_quantiles: int = config.model.num_quantiles
        self.num_cosines: int = config.model.num_cosines
        self.state_embedder = MLP(
            config=config, input_size=self.embedding_dim, output_size=self.embedding_dim
        )
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=self.num_cosines, embedding_dim=self.embedding_dim
        )
        self.fc_V = MLP(
            config=config, input_size=self.num_quantiles * self.embedding_dim, output_size=self.num_quantiles
        )
        self.fc_A = MLP(
            config=config,
            input_size=self.num_quantiles * self.embedding_dim,
            output_size=output_size * self.num_quantiles
        )

    def forward(self, state, external_taus: torch.Tensor = None):
        state_embeddings = self.get_state_embeddings(state=state)
        quantiles = self.get_quantiles(state_embeddings=state_embeddings, external_taus=external_taus)
        q_values = quantiles.mean(dim=2)

        return q_values

    def state_embeddings(self, state):
        x = self.state_encoder(state)

        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        state_embeddings = self.state_embedder(out)

        return state_embeddings

    def get_taus(self, device):
        taus = torch.rand(1, self.num_quantiles, device=device)

        return taus

    def get_quantiles(self, state_embeddings, external_taus: torch.Tensor = None):
        taus = external_taus or self.get_taus(device=state_embeddings.device)
        tau_embeddings = self.cosine_net(taus)

        # Reshape into (batch_size, 1, embedding_dim).
        batch_size = state_embeddings.shape[0]
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

    def get_mlp_input_size(self, input_shape: List[int]) -> int:
        random_input = torch.randn(size=input_shape).unsqueeze(0)
        outputs = self.conv(random_input)

        return outputs.view(-1).shape[0]

    def state_encoder(self, state):
        return state[self.view_method]
