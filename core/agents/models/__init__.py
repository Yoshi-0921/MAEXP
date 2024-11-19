"""Builds network used in brain of learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from omegaconf import DictConfig
from torch import nn

from core.utils.logging import initialize_logging

from .customs.ada3_iqn import ADA3_IQN, MergedADA3_IQN
from .customs.categorical_dqn import CategoricalDQN
from .customs.cda3 import CDA3
from .customs.cda3_iqn import CDA3_IQN
from .customs.conv_mlp import ConvMLP, MergedConvMLP
from .customs.da3 import DA3, MergedDA3
from .customs.da3_iqn import DA3_IQN, MergedDA3_IQN
from .customs.da6 import DA6
from .customs.da6_iqn import DA6_IQN
from .customs.da6_iqn_cond import DA6_IQN_Cond
from .customs.dra3_dqn import DRA3_DQN
from .customs.dra3_iqn import DRA3_IQN
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
        if config.observation_area_mask == "merged":
            network = MergedConvMLP(config=config, input_shape=obs_shape, output_size=act_size)
        else:
            network = ConvMLP(config=config, input_shape=obs_shape, output_size=act_size)
    elif config.model.name == "da3":
        if config.observation_area_mask == "merged":
            network = MergedDA3(config=config, input_shape=obs_shape, output_size=act_size)
        else:
            network = DA3(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "da3_iqn":
        if config.observation_area_mask == "merged":
            network = MergedDA3_IQN(config=config, input_shape=obs_shape, output_size=act_size)
        else:
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

    elif config.model.name == "da6_iqn":
        network = DA6_IQN(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "da6_iqn_cond":
        network = DA6_IQN_Cond(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "dra3_iqn":
        network = DRA3_IQN(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "dra3_dqn":
        network = DRA3_DQN(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "cda3":
        network = CDA3(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "cda3_iqn":
        network = CDA3_IQN(config=config, input_shape=obs_shape, output_size=act_size)

    elif config.model.name == "ada3_iqn":
        if config.observation_area_mask == "merged":
            network = MergedADA3_IQN(config=config, input_shape=obs_shape, output_size=act_size)
        else:
            network = ADA3_IQN(config=config, input_shape=obs_shape, output_size=act_size)

    else:
        logger.warn(f"Unexpected network is given. config.model.name: {config.model.name}")

        raise ValueError()

    return network
