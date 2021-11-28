"""Builds network used in brain of learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn
from typing import List
from .customs.conv_mlp import ConvMLP
from .mlp import MLP
from .customs.da3 import DA3
from .customs.da6 import DA6
from .customs.categorical_dqn import CategoricalDQN
from .customs.quantile_regression_dqn import QRDQN

logger = initialize_logging(__name__)


def generate_network(
    config: DictConfig, obs_shape: List[int], act_size: int
) -> nn.Module:
    if config.model.name == "mlp":
        network = MLP(config=config, input_size=obs_shape[0], output_size=act_size)

    elif config.model.name == "conv_mlp":
        network = ConvMLP(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "da3":
        network = DA3(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "da6":
        network = DA6(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "categorical_dqn":
        network = CategoricalDQN(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "qr_dqn":
        network = QRDQN(config=config, input_shape=obs_shape, output_size=act_size)

    else:
        logger.warn(f"Unexpected network is given. config.network: {config.network}")

        raise ValueError()

    return network
