# -*- coding: utf-8 -*-

"""Source code for simple multi-perceptron layer class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

logger = initialize_logging(__name__)


class MLP(nn.Module):
    def __init__(self, config: DictConfig, input_size: int, output_size: int):
        super().__init__()
        self.config = config

        self.af_list = self.add_activation_functions()

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

    def add_activation_functions(self):
        af_list = []
        for activation_function in self.config.model.activation_functions:
            if activation_function == 'relu':
                af_list.append(F.relu)

            elif activation_function == 'sigmoid':
                af_list.append(F.sigmoid)

            elif activation_function == 'tanh':
                af_list.append(F.tanh)

            else:
                logger.warn(f"Unexpected activation function is given. activation_function: {activation_function}")

                raise ValueError()

        return af_list
