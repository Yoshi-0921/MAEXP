"""Source code for implicit quantile network (IQN) layers class.

Original code: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
"""

from typing import List

import torch
from omegaconf import DictConfig

from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging

from ..convolution import Conv
from ..mlp import MLP
from .iqn import IQN, CosineEmbeddingNetwork

logger = initialize_logging(__name__)


class IQN_Cond(IQN):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__(config=config, input_shape=input_shape, output_size=output_size)
        self.local_conv = Conv(
            config=config,
            input_channel=input_shape[0],
            output_channel=config.model.output_channel,
        )
        self.relative_conv = Conv(
            config=config,
            input_channel=1,
            output_channel=config.model.output_channel // 2,
        )
        self.destination_conv = Conv(
            config=config,
            input_channel=1,
            output_channel=config.model.output_channel // 2,
        )

        local_embedding_dim: int = IQN_Cond.get_mlp_input_size(self.local_conv, [input_shape[0], config.visible_range, config.visible_range])
        relative_embedding_dim: int = IQN_Cond.get_mlp_input_size(self.relative_conv, [1, *input_shape[1:]])
        destination_embedding_dim: int = IQN_Cond.get_mlp_input_size(self.destination_conv, [1, *input_shape[1:]])

        self.embedding_dim: int = local_embedding_dim + relative_embedding_dim + destination_embedding_dim
        self.local_state_embedder = MLP(
            config=config, input_size=local_embedding_dim, output_size=local_embedding_dim
        )
        self.relative_state_embedder = MLP(
            config=config, input_size=relative_embedding_dim, output_size=relative_embedding_dim
        )
        self.destination_state_embedder = MLP(
            config=config, input_size=relative_embedding_dim, output_size=relative_embedding_dim
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

    def get_state_embeddings(self, state):
        local_x, relative_x, destination_x = self.state_encoder(state)

        relative_out = self.relative_conv(relative_x)
        relative_out = relative_out.view(relative_out.shape[0], -1)
        relative_state_embeddings = self.relative_state_embedder(relative_out)

        destination_out = self.destination_conv(destination_x)
        destination_out = destination_out.view(destination_out.shape[0], -1)
        destination_embeddings = self.destination_state_embedder(destination_out)

        local_out = self.local_conv(local_x)
        local_out = local_out.view(local_out.shape[0], -1)
        local_state_embeddings = self.local_state_embedder(local_out)

        state_embeddings = torch.cat((relative_state_embeddings, destination_embeddings, local_state_embeddings), dim=1)

        return state_embeddings

    def state_encoder(self, state):
        local_x = state["local"]
        # x.shape: [1, 4, 25, 25]
        relative_x = ObservationHandler.decode_relative_state(
            state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
        )
        relative_x = relative_x[:, -1:, ...]  # [1, 1, 25, 25]が欲しい
        relative_x += 1

        destination_x = state["destination"].unsqueeze(1)

        return local_x, relative_x, destination_x
