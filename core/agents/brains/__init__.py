# -*- coding: utf-8 -*-

"""Builds brain used in learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_brain import AbstractBrain
from .da6_brain import DA6Brain, DA6BaselineBrain
from .dqn_brain import DQNBrain
from .mat_brain import MATBaselineBrain, MATBrain

logger = initialize_logging(__name__)

__all__ = ["AbstractBrain", "DQNBrain", "MATBrain", "DA6Brain"]


def generate_brain(
    config: DictConfig, obs_shape: List[int], act_size: int
) -> AbstractBrain:
    if config.brain == "dqn":
        brain = DQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "mat":
        brain = MATBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "mat_baseline":
        brain = MATBaselineBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da6":
        brain = DA6Brain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da6_baseline":
        brain = DA6BaselineBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    else:
        logger.warn(f"Unexpected brain is given. config.brain: {config.brain}")

        raise ValueError()

    return brain
