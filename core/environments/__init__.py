# -*- coding: utf-8 -*-

"""Builds environments used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig

from .abstract_environment import AbstractEnvironment
from .default_environment import DefaultEnvironment


logger = initialize_logging(__name__)

__all__ = ["AbstractEnvironment", "DefaultEnvironment"]


def generate_environment(config: DictConfig, world: AbstractWorld) -> AbstractEnvironment:
    if config.environment == "default":
        env = DefaultEnvironment(config=config, world=world)

        return env

    else:
        logger.warn(f"Unexpected environment is given. config.environment: {config.environment}")

        raise ValueError()


def generate_test_environment(config: DictConfig) -> AbstractEnvironment:
    raise NotImplementedError()
