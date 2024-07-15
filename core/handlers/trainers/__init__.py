"""Builds trainer to execute multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from omegaconf import DictConfig

from core.environments.abstract_environment import AbstractEnvironment
from core.utils.logging import initialize_logging

from .abstract_trainer import AbstractTrainer
from .attention_trainer import AttentionTrainer
from .default_trainer import DefaultTrainer
from .recurrent_attention_trainer import RecurrentAttentionTrainer
from .recurrent_trainer import RecurrentTrainer

logger = initialize_logging(__name__)


def generate_trainer(
    config: DictConfig, environment: AbstractEnvironment
) -> AbstractTrainer:
    if config.trainer == "default":
        trainer = DefaultTrainer(config=config, environment=environment)

    elif config.trainer == "attention":
        trainer = AttentionTrainer(config=config, environment=environment)

    elif config.trainer == "recurrent":
        trainer = RecurrentTrainer(config=config, environment=environment)

    elif config.trainer == "recurrent_attention":
        trainer = RecurrentAttentionTrainer(config=config, environment=environment)

    else:
        logger.warn(f"Unexpected trainer is given. config.trainer: {config.trainer}")

        raise ValueError()

    return trainer
