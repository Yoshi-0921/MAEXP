# -*- coding: utf-8 -*-

"""Builds observation mask handler.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig

from .default_mask import generate_default_mask

logger = initialize_logging(__name__)


def generate_observation_mask(config: DictConfig, world: AbstractWorld):
    if config.observation_mask == 'default':
        mask = generate_default_mask(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected mask generator is given. config.observation_mask: {config.observation_mask}"
        )

        raise ValueError()

    return mask
