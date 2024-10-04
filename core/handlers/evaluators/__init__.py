"""Builds evaluator to execute multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from omegaconf import DictConfig

from core.environments.abstract_environment import AbstractEnvironment
from core.utils.logging import initialize_logging

from .abstract_evaluator import AbstractEvaluator
from .attention_evaluator import AttentionEvaluator
from .attention_types_evaluator import AttentionWanderingEvaluator
from .da3_types_video_evaluator import DA3WanderingVideoEvaluator
from .da3_video_evaluator import DA3VideoEvaluator
from .default_evaluator import DefaultEvaluator

logger = initialize_logging(__name__)


def generate_evaluator(
    config: DictConfig, environment: AbstractEnvironment
) -> AbstractEvaluator:
    if config.evaluator == "default":
        evaluator = DefaultEvaluator(config=config, environment=environment)

    elif config.evaluator == "attention":
        evaluator = AttentionEvaluator(config=config, environment=environment)

    elif config.evaluator == "attention_wandering":
        assert config.num_agents == 6
        evaluator = AttentionWanderingEvaluator(config=config, environment=environment)

    elif config.evaluator == "da3_video":
        evaluator = DA3VideoEvaluator(config=config, environment=environment)

    elif config.evaluator == "da3_wandering_video":
        evaluator = DA3WanderingVideoEvaluator(config=config, environment=environment)

    else:
        logger.warn(
            f"Unexpected evaluator is given. config.evaluator: {config.evaluator}"
        )

        raise ValueError()

    return evaluator
