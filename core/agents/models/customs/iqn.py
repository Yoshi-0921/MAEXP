"""Source code for implicit quantile network (IQN) layers class.

Original code: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
"""

from typing import List

import torch
from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from ..convolution import Conv
from ..mlp import MLP

logger = initialize_logging(__name__)


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU(inplace=False))
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = torch.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device
        ).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(
            batch_size * N, self.num_cosines
        )

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)

        return tau_embeddings


class IQN(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.view_method = config.observation_area_mask
        self.map_SIZE_X, self.map_SIZE_Y = config.map.SIZE_X, config.map.SIZE_Y
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
            config=config,
            input_size=self.embedding_dim,
            output_size=1
        )
        self.fc_A = MLP(
            config=config,
            input_size=self.embedding_dim,
            output_size=output_size,
        )

    def forward(self, state, external_taus: torch.Tensor = None):
        state_embeddings = self.get_state_embeddings(state=state)
        taus = external_taus if external_taus is not None else self.get_taus(state_embeddings=state_embeddings)
        quantiles = self.get_quantiles(state_embeddings=state_embeddings, taus=taus)
        q_values = self.get_q_values(quantiles=quantiles)

        return q_values

    def get_state_embeddings(self, state):
        x = self.state_encoder(state)

        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        state_embeddings = self.state_embedder(out)

        return state_embeddings

    def get_taus(self, state_embeddings):
        return torch.rand(1, self.num_quantiles, device=state_embeddings.device)

    def get_quantiles(
        self,
        state=None,
        state_embeddings: torch.Tensor = None,
        taus: torch.Tensor = None,
    ):
        state_embeddings = state_embeddings if state_embeddings is not None else self.get_state_embeddings(state)
        tau_embeddings = self.cosine_net(taus)

        # Reshape into (batch_size, 1, embedding_dim).
        batch_size = state_embeddings.shape[0]
        state_embeddings = state_embeddings.view(batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        num_quantiles = tau_embeddings.shape[1]
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * num_quantiles, self.embedding_dim
        )

        V = self.fc_V(embeddings)
        A = self.fc_A(embeddings)

        V = V.view(batch_size, num_quantiles, 1)  # .transpose(1, 2)
        A = A.view(batch_size, num_quantiles, self.num_actions)  # .transpose(1, 2)

        average_A = A.mean(2, keepdim=True)
        quantiles = V.expand_as(A) + (A - average_A.expand_as(A))

        return quantiles

    def get_q_values(self, quantiles: torch.Tensor):
        return quantiles.mean(dim=1)

    def get_mlp_input_size(self, input_shape: List[int]) -> int:
        random_input = torch.randn(size=input_shape).unsqueeze(0)
        outputs = self.conv(random_input)

        return outputs.view(-1).shape[0]

    def state_encoder(self, state):
        if self.view_method == "relative":
            return ObservationHandler.decode_relative_state(
                state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
            )

        return state[self.view_method]
