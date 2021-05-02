# -*- coding: utf-8 -*-

"""Builds trainer to execute multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_environment import AbstractEnvironment

logger = initialize_logging(__name__)


def generate_trainer(config: DictConfig, environment: AbstractEnvironment):
    if config.trainer == "default":
        trainer = None

        return trainer

    else:
        logger.warn(f"Unexpected trainer is given. config.trainer: {config.trainer}")

        raise ValueError()


def generate_evaluator(config: DictConfig):
    raise NotImplementedError()
