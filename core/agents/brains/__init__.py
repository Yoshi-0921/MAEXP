# -*- coding: utf-8 -*-

"""Builds brain used in learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_brain import AbstractBrain
from .dqn_brain import DQNBrain

logger = initialize_logging(__name__)

__all__ = ["AbstractBrain", "DQNBrain"]


def generate_brain(
    config: DictConfig, obs_shape: List[int], act_size: int
) -> AbstractBrain:
    if config.brain == "dqn":
        brain = DQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

        return brain

    else:
        logger.warn(f"Unexpected brain is given. config.brain: {config.brain}")

        raise ValueError()
