# -*- coding: utf-8 -*-

"""Source code for simple multi-perceptron layer class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from omegaconf import DictConfig
from torch import nn

from .activation_functions import add_activation_functions


class MLP(nn.Module):
    def __init__(self, config: DictConfig, input_size: int, output_size: int):
        super().__init__()
        self.config = config

        self.af_list = add_activation_functions(self.config.model.activation_functions)

        self.fc_list = nn.ModuleList()
        for layer_size in config.model.hidden_layer_sizes:
            self.fc_list.append(nn.Linear(input_size, layer_size))
            input_size = layer_size

        self.fc_post = nn.Linear(input_size, output_size)

    def forward(self, x):
        for af, fc in zip(self.af_list, self.fc_list):
            x = af(fc(x))

        outputs = self.fc_post(x)

        return outputs
