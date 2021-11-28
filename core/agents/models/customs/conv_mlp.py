"""Source code for convolution-mlp layers class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from ..convolution import Conv
from ..mlp import MLP

logger = initialize_logging(__name__)


class ConvMLP(nn.Module):
    def __init__(self, config: DictConfig, input_shape: List[int], output_size: int):
        super().__init__()
        self.conv = Conv(
            config=config,
            input_channel=input_shape[0],
            output_channel=config.model.output_channel,
        )
        input_size = self.get_mlp_input_size(input_shape)
        self.mlp = MLP(config=config, input_size=input_size, output_size=output_size)

    def forward(self, x):
        out = self.conv(x)

        out = out.view(out.shape[0], -1)

        outputs = self.mlp(out)

        return outputs

    def get_mlp_input_size(self, input_shape: List[int]):
        random_input = torch.randn(size=input_shape).unsqueeze(0)
        outputs = self.conv(random_input)

        return outputs.view(-1).shape[0]
