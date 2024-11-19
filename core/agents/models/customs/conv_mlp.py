"""Source code for convolution-mlp layers class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from core.handlers.observations.observation_handler import ObservationHandler
from core.utils.logging import initialize_logging

from ..convolution import Conv
from ..mlp import MLP

logger = initialize_logging(__name__)


class ConvMLP(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.view_method = config.observation_area_mask
        self.objects_channel = config.objects_channel
        in_chans = input_shape[0]
        if self.objects_channel:
            in_chans += config.type_objects
        self.conv = Conv(
            config=config,
            input_channel=in_chans,
            output_channel=config.model.output_channel,
        )
        input_size = ConvMLP.get_mlp_input_size(self.conv, [in_chans, *input_shape[1:]])
        self.mlp = MLP(config=config, input_size=input_size, output_size=output_size)
        self.map_SIZE_X, self.map_SIZE_Y = config.map.SIZE_X, config.map.SIZE_Y

    def forward(self, state):
        x = self.state_encoder(state)

        out = self.conv(x)

        out = out.view(out.shape[0], -1)

        outputs = self.mlp(out)

        return outputs

    @staticmethod
    def get_mlp_input_size(conv, input_shape: List[int]) -> int:
        random_input = torch.randn(size=input_shape).unsqueeze(0)
        outputs = conv(random_input)

        return outputs.view(-1).shape[0]

    def state_encoder(self, state):
        if self.view_method == "relative":
            relative_x = ObservationHandler.decode_relative_state(
                state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
            )
            if self.objects_channel:
                relative_x = torch.cat((relative_x, state["objects"]), dim=1)

            return relative_x

        return state[self.view_method]


class MergedConvMLP(ConvMLP):
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
            output_channel=config.model.output_channel,
        )
        local_embedding_dim: int = MergedConvMLP.get_mlp_input_size(self.local_conv, [input_shape[0], config.visible_range, config.visible_range])
        relative_embedding_dim: int = MergedConvMLP.get_mlp_input_size(self.relative_conv, [1, *input_shape[1:]])
        self.mlp = MLP(config=config, input_size=local_embedding_dim+relative_embedding_dim, output_size=output_size)

    def forward(self, state):
        local_x, relative_x = self.state_encoder(state)

        relative_out = self.relative_conv(relative_x)
        relative_state_embeddings = relative_out.view(relative_out.shape[0],-1)
        local_out = self.local_conv(local_x)
        local_state_embeddings = local_out.view(local_out.shape[0],-1)
        state_embeddings = torch.cat((relative_state_embeddings, local_state_embeddings), dim=1)

        outputs = self.mlp(state_embeddings)

        return outputs

    def state_encoder(self, state):
        local_x = state["local"]
        # x.shape: [1, 4, 25, 25]
        relative_x = ObservationHandler.decode_relative_state(
            state=state, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
        )
        relative_x = relative_x[:, -1:, ...]  # [1, 1, 25, 25]が欲しい
        relative_x += 1
        return local_x, relative_x
