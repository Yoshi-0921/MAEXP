"""Builds trainer to execute multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.environments.abstract_environment import AbstractEnvironment
from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_trainer import AbstractTrainer
from .default_trainer import DefaultTrainer
from .attention_trainer import AttentionTrainer
from .attention_types_trainer import AttentionWanderingTrainer

logger = initialize_logging(__name__)


def generate_trainer(
    config: DictConfig, environment: AbstractEnvironment
) -> AbstractTrainer:
    if config.trainer == "default":
        trainer = DefaultTrainer(config=config, environment=environment)

    elif config.trainer == "attention":
        trainer = AttentionTrainer(config=config, environment=environment)

    elif config.trainer == "attention_wandering":
        assert config.num_agents == 6
        trainer = AttentionWanderingTrainer(config=config, environment=environment)

    else:
        logger.warn(f"Unexpected trainer is given. config.trainer: {config.trainer}")

        raise ValueError()

    return trainer
