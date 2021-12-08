"""Source code for categorical dqn (C51) layers class.

Original code: https://github.com/Kaixhin/Rainbow
"""

from enum import IntEnum
from typing import List

import torch
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from ..convolution import Conv
from ..mlp import MLP

logger = initialize_logging(__name__)


class ApplySoftmax(IntEnum):
    NONE = 0
    NORMAL = 1
    LOG = 2


class CategoricalDQN(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.view_method = config.observation_area_mask
        self.num_actions = output_size
        self.conv = Conv(
            config=config,
            input_channel=input_shape[0],
            output_channel=config.model.output_channel,
        )
        input_size: int = self.get_mlp_input_size(input_shape)
        self.num_atoms: int = config.model.num_atoms
        self.fc_V = MLP(
            config=config, input_size=input_size, output_size=self.num_atoms
        )
        self.fc_A = MLP(
            config=config, input_size=input_size, output_size=output_size * self.num_atoms
        )

    def forward(self, state, apply_softmax=ApplySoftmax.NONE):
        x = self.state_encoder(state)

        out = self.conv(x)

        out = out.view(out.shape[0], -1)

        V = self.fc_V(out)
        A = self.fc_A(out)

        V = V.view(out.shape[0], 1, self.num_atoms)
        A = A.view(out.shape[0], self.num_actions, self.num_atoms)

        average_A = A.mean(1, keepdim=True)
        outputs = V.expand_as(A) + (A - average_A.expand_as(A))

        if apply_softmax == ApplySoftmax.NORMAL:
            return F.softmax(outputs, dim=2)

        if apply_softmax == ApplySoftmax.LOG:
            return F.log_softmax(outputs, dim=2)

        return outputs

    def get_mlp_input_size(self, input_shape: List[int]) -> int:
        random_input = torch.randn(size=input_shape).unsqueeze(0)
        outputs = self.conv(random_input)

        return outputs.view(-1).shape[0]

    def state_encoder(self, state):

        return state[self.view_method]
