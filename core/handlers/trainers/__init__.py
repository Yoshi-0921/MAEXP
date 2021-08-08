# -*- coding: utf-8 -*-

"""Builds trainer to execute multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.environments.abstract_environment import AbstractEnvironment
from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_trainer import AbstractTrainer
from .default_trainer import DefaultTrainer
from .mat_trainer import MATTrainer
from .mat_types_trainer import MATTypesTrainer

logger = initialize_logging(__name__)


def generate_trainer(
    config: DictConfig, environment: AbstractEnvironment
) -> AbstractTrainer:
    if config.trainer == "default":
        trainer = DefaultTrainer(config=config, environment=environment)

    elif config.trainer == "mat":
        trainer = MATTrainer(config=config, environment=environment)

    elif config.trainer == "mat_types":
        trainer = MATTypesTrainer(config=config, environment=environment)

    else:
        logger.warn(f"Unexpected trainer is given. config.trainer: {config.trainer}")

        raise ValueError()

    return trainer
