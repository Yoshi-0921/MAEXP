# -*- coding: utf-8 -*-

"""Builds observation mask handler.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig
import torch
from .default_area_mask import generate_default_area_mask

logger = initialize_logging(__name__)


def generate_observation_area_mask(
    config: DictConfig, world: AbstractWorld
) -> torch.Tensor:
    if config.observation_area_mask == "default":
        area_mask = generate_default_area_mask(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected area mask generator is given. config.observation_area_mask: {config.observation_area_mask}"
        )

        raise ValueError()

    return area_mask
