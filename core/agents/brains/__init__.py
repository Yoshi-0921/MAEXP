"""Builds brain used in learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from omegaconf import DictConfig

from core.utils.logging import initialize_logging

from .abstract_brain import AbstractBrain
from .categorical_dqn_brain import CategoricalDQNBrain
from .da3_brain import DA3BaselineBrain, DA3Brain
from .da3_iqn_brain import DA3_IQNBrain
from .da6_brain import DA6Brain
from .dqn_brain import DQNBrain
from .dra3_dqn_brain import DRA3_DQNBrain
from .dra3_iqn_brain import DRA3_IQNBrain
from .fqf_brain import FQFBrain
from .iqn_brain import IQNBrain
from .qr_dqn_brain import QRDQNBrain

logger = initialize_logging(__name__)

__all__ = [
    "AbstractBrain",
    "DQNBrain",
    "DA3Brain",
    "DA3BaselineBrain",
    "DRA3_IQNBrain",
    "DA6Brain",
    "CategoricalDQNBrain",
    "QRDQNBrain",
    "IQNBrain",
    "DA3_IQNBrain",
    "FQFBrain",
    "DRA3_DQNBrain"
]


def generate_brain(
    config: DictConfig, obs_shape: List[int], act_size: int
) -> AbstractBrain:
    if config.brain == "dqn":
        brain = DQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da3":
        brain = DA3Brain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da3_iqn":
        brain = DA3_IQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da3_baseline":
        brain = DA3BaselineBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "dra3_dqn":
        brain = DRA3_DQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "dra3_iqn":
        brain = DRA3_IQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "da6":
        brain = DA6Brain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "categorical_dqn":
        brain = CategoricalDQNBrain(
            config=config, obs_shape=obs_shape, act_size=act_size
        )

    elif config.brain == "qr_dqn":
        brain = QRDQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    elif config.brain == "iqn":
        brain = IQNBrain(config=config, obs_shape=obs_shape, act_size=act_size)
    
    elif config.brain == "fqf":
        brain = FQFBrain(config=config, obs_shape=obs_shape, act_size=act_size)

    else:
        logger.warn(f"Unexpected brain is given. config.brain: {config.brain}")

        raise ValueError()

    return brain
