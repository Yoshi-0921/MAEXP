# -*- coding: utf-8 -*-

"""Builds network used in brain of learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from .customs.conv_mlp import ConvMLP
from .customs.mast import MAST
from .customs.mat import MAT
from .mlp import MLP

logger = initialize_logging(__name__)


def generate_network(config: DictConfig, obs_shape: List[int], act_size: int) -> nn.Module:
    if config.model.name == 'mlp':
        network = MLP(config=config, input_size=obs_shape[0], output_size=act_size)

    elif config.model.name in ['conv_mlp', 'mat_baseline']:
        network = ConvMLP(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == 'mat':
        network = MAST(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == 'mast':
        network = MAST(config=config, input_shape=obs_shape, output_size=act_size)

    else:
        logger.warn(f"Unexpected network is given. config.network: {config.network}")

        raise ValueError()

    return network
