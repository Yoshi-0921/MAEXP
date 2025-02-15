# -*- coding: utf-8 -*-

"""Builds environments used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from omegaconf import DictConfig

from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld

from .abstract_environment import AbstractEnvironment
from .default_environment import DefaultEnvironment
from .observation_stats_environment import ObservationStatsEnvironment
from .sequential_environment import SequentialEnvironment
from .junctional_environment import JunctionalEnvironment
from .shared_reward_environment import (SharedMaxRewardEnvironment,
                                        SharedMeanRewardEnvironment)
from .test_environment import TestEnvironment
from .test_sequential_environment import TestSequentialEnvironment

logger = initialize_logging(__name__)

__all__ = [
    "AbstractEnvironment",
    "DefaultEnvironment",
    "SharedMaxRewardEnvironment",
    "SharedMeanRewardEnvironment",
    "TypesEnvironment",
    "TestEnvironment",
    "SequentialEnvironment",
    "JunctionalEnvironment"
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

    elif config.environment == 'test':
        env = TestEnvironment(config=config, world=world)

    elif config.environment == "observation_stats":
        env = ObservationStatsEnvironment(config=config, world=world)

    elif config.environment == "sequential":
        env = SequentialEnvironment(config=config, world=world)

    elif config.environment == "junctional":
        env = JunctionalEnvironment(config=config, world=world)

    elif config.environment == "sequential_test":
        env = TestSequentialEnvironment(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected environment is given. config.environment: {config.environment}"
        )

        raise ValueError()

    return env
