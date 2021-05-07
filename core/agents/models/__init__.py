# -*- coding: utf-8 -*-

"""Builds network used in brain of learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import nn

from .mlp import MLP

logger = initialize_logging(__name__)


def generate_network(config: DictConfig, obs_size: int, act_size: int) -> nn.Module:
    network = MLP(config=config, input_size=obs_size, output_size=act_size)

    return network
