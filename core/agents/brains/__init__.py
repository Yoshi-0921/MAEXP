# -*- coding: utf-8 -*-

"""Builds brain used in learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .dqn_brain import DQNBrain
from .abstract_brain import AbstractBrain

logger = initialize_logging(__name__)

__all__ = ["AbstractBrain", "DQNBrain"]


def generate_brain(config: DictConfig, obs_size: int, act_size: int) -> AbstractBrain:
    if config.brain == "dqn":
        brain = DQNBrain(config=config, obs_size=obs_size, act_size=act_size)

        return brain

    else:
        logger.warn(f"Unexpected brain is given. config.brain: {config.brain}")

        raise ValueError()
