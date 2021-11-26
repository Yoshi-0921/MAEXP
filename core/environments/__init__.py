# -*- coding: utf-8 -*-

"""Builds environments used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig

from .abstract_environment import AbstractEnvironment
from .default_environment import DefaultEnvironment
from .shared_reward_environment import (
    SharedMaxRewardEnvironment,
    SharedMeanRewardEnvironment,
)
from .types_environment import TypesEnvironment
from .types_test_environment import TypesTestEnvironment

logger = initialize_logging(__name__)

__all__ = [
    "AbstractEnvironment",
    "DefaultEnvironment",
    "SharedMaxRewardEnvironment",
    "SharedMeanRewardEnvironment",
    "TypesEnvironment"
]


def generate_environment(
    config: DictConfig, world: AbstractWorld
) -> AbstractEnvironment:
    if config.environment == "default":
        env = DefaultEnvironment(config=config, world=world)

    elif config.environment == "shared_mean_reward":
        env = SharedMeanRewardEnvironment(config=config, world=world)

    elif config.environment == "shared_max_reward":
        env = SharedMaxRewardEnvironment(config=config, world=world)

    elif config.environment == 'object_types':
        env = TypesEnvironment(config=config, world=world)

    elif config.environment == 'da3_types_test':
        env = TypesTestEnvironment(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected environment is given. config.environment: {config.environment}"
        )

        raise ValueError()

    return env
