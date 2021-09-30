# -*- coding: utf-8 -*-

"""Builds evaluator to execute multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.environments.abstract_environment import AbstractEnvironment
from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_evaluator import AbstractEvaluator
from .default_evaluator import DefaultEvaluator
from .mat_evaluator import MATEvaluator
from .mat_types_evaluator import MATTypesEvaluator
from .mat_types_video_evaluator import MATTypesVideoEvaluator

logger = initialize_logging(__name__)


def generate_evaluator(
    config: DictConfig, environment: AbstractEnvironment
) -> AbstractEvaluator:
    if config.evaluator == "default":
        evaluator = DefaultEvaluator(config=config, environment=environment)

    elif config.evaluator == "mat":
        evaluator = MATEvaluator(config=config, environment=environment)

    elif config.evaluator == "mat_types":
        evaluator = MATTypesEvaluator(config=config, environment=environment)

    elif config.evaluator == "mat_types_video":
        evaluator = MATTypesVideoEvaluator(config=config, environment=environment)

    else:
        logger.warn(
            f"Unexpected evaluator is given. config.evaluator: {config.evaluator}"
        )

        raise ValueError()

    return evaluator
