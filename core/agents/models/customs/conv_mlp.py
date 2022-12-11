"""Source code for convolution-mlp layers class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
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
        input_size = self.get_mlp_input_size([in_chans, *input_shape[1:]])
        self.mlp = MLP(config=config, input_size=input_size, output_size=output_size)
        self.map_SIZE_X, self.map_SIZE_Y = config.map.SIZE_X, config.map.SIZE_Y

    def forward(self, state):
        x = self.state_encoder(state)

        out = self.conv(x)

        out = out.view(out.shape[0], -1)

        outputs = self.mlp(out)

        return outputs

    def get_mlp_input_size(self, input_shape: List[int]):
        random_input = torch.randn(size=input_shape).unsqueeze(0)
        outputs = self.conv(random_input)

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
