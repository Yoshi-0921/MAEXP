# -*- coding: utf-8 -*-

"""Builds environments used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_environment import AbstractEnvironment
from .tools.entity import Agent, Object
from .tools.world import Map, World

logger = initialize_logging(__name__)

__all__ = ["Agent", "Object", "Map", "World"]


def generate_environment(config: DictConfig) -> AbstractEnvironment:
    if config.environment == "default":
        env = None

        return env

    else:
        logger.warn(f"Unexpected environment is given. config.environment: {config.environment}")

        raise ValueError()


def generate_test_environment(config: DictConfig) -> AbstractEnvironment:
    raise NotImplementedError()
