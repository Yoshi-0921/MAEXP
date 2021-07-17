# -*- coding: utf-8 -*-

"""Builds network used in brain of learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn
from typing import List
from .customs.conv_mlp import ConvMLP
from .mlp import MLP
from .customs.mat import MAT

logger = initialize_logging(__name__)


def generate_network(
    config: DictConfig, obs_shape: List[int], act_size: int
) -> nn.Module:
    if config.model.name == "mlp":
        network = MLP(config=config, input_size=obs_shape[0], output_size=act_size)

    elif config.model.name in ["conv_mlp", "mat_baseline"]:
        network = ConvMLP(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "mat":
        network = MAT(config=config, input_shape=obs_shape, output_size=act_size)

    else:
        logger.warn(f"Unexpected network is given. config.network: {config.network}")

        raise ValueError()

    return network
