# -*- coding: utf-8 -*-

"""Source code for simple convolution layer class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from omegaconf import DictConfig
from torch import nn

from .activation_functions import add_activation_functions


class Conv(nn.Module):
    def __init__(self, config: DictConfig, input_channel: int, output_channel: int):
        super().__init__()
        self.config = config

        self.af_list = add_activation_functions(config.model.conv_afs)

        self.conv_list = nn.ModuleList()
        for channel_size, kernel_size, stride in zip(
            config.model.hidden_channels,
            config.model.hidden_kernels,
            config.model.hidden_strides,
        ):
            self.conv_list.append(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=channel_size,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            input_channel = channel_size

        self.conv_post = nn.Conv2d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=config.model.hidden_kernels[-1],
            stride=config.model.hidden_strides[-1],
        )

    def forward(self, x):
        for af, kernel_size, stride, conv in zip(
            self.af_list,
            self.config.model.af_kernels,
            self.config.model.af_strides,
            self.conv_list,
        ):
            x = af(conv(x), kernel_size=kernel_size, stride=stride)

        outputs = self.conv_post(x)

        return outputs
