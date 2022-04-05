"""Builds network used in brain of learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from .customs.categorical_dqn import CategoricalDQN
from .customs.conv_mlp import ConvMLP
from .customs.da3 import DA3
from .customs.da3_iqn import DA3_IQN
from .customs.da6 import DA6
from .customs.fqf import FQF
from .customs.iqn import IQN, MergedIQN
from .customs.qr_dqn import QRDQN
from .mlp import MLP

logger = initialize_logging(__name__)


def generate_network(
    config: DictConfig, obs_shape: List[int], act_size: int, target: bool = False
) -> nn.Module:
    if config.model.name == "mlp":
        network = MLP(config=config, input_size=obs_shape[0], output_size=act_size)

    elif config.model.name in ["conv_mlp", "da3_baseline"]:
        network = ConvMLP(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "da3":
        network = DA3(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "da3_iqn":
        network = DA3_IQN(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "da6":
        network = DA6(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "categorical_dqn":
        network = CategoricalDQN(
            config=config, input_shape=obs_shape, output_size=act_size
        )

    elif config.model.name == "qr_dqn":
        network = QRDQN(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "iqn":
        if config.observation_area_mask == "merged":
            network = MergedIQN(config=config, input_shape=obs_shape, output_size=act_size)
        else:
            network = IQN(config=config, input_shape=obs_shape, output_size=act_size)
    
    elif config.model.name == "fqf":
        network = FQF(config=config, input_shape=obs_shape, output_size=act_size, target=target)

    else:
        logger.warn(f"Unexpected network is given. config.model.name: {config.model.name}")

        raise ValueError()

    return network
