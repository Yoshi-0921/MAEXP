"""Builds brain used in learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_brain import AbstractBrain
from .da3_brain import DA3BaselineBrain, DA3Brain
from .da6_brain import DA6Brain
from .dqn_brain import DQNBrain

logger = initialize_logging(__name__)

__all__ = ["AbstractBrain", "DQNBrain", "DA3Brain", "DA3BaselineBrain", "DA6Brain"]


def generate_brain(
    config: DictConfig, obs_shape: List[int], act_size: int
) -> AbstractBrain:
    if config.brain == "dqn":
        brain = DQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da3":
        brain = DA3Brain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da3_baseline":
        brain = DA3BaselineBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da6":
        brain = DA6Brain(config=config, obs_shape=obs_shape, act_size=act_size)

    else:
        logger.warn(f"Unexpected brain is given. config.brain: {config.brain}")

        raise ValueError()

    return brain
