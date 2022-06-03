# -*- coding: utf-8 -*-

"""Builds observation mask handler.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig

from .abstract_observation_coordinate_handler import (
    AbstractObservationCoordinateHandler,
)
from .local_area_mask import LocalObservationCoordinateHandler, generate_local_area_mask
from .relative_area_mask import (
    RelativeObservationCoordinateHandler,
    generate_relative_area_mask,
)

logger = initialize_logging(__name__)


def generate_observation_area_mask(
    config: DictConfig, world: AbstractWorld
) -> torch.Tensor:
    if config.observation_area_mask == "local":
        area_mask = generate_local_area_mask(config=config, world=world)

    elif config.observation_area_mask == "relative":
        area_mask = generate_local_area_mask(config=config, world=world)
        # area_mask = generate_relative_area_mask(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected area mask generator is given. config.observation_area_mask: {config.observation_area_mask}"
        )

        raise ValueError()

    return area_mask


def generate_observation_mask_coordinate(
    config: DictConfig, world: AbstractWorld
) -> AbstractObservationCoordinateHandler:
    if config.observation_area_mask == "local":
        observation_mask_coordinate = LocalObservationCoordinateHandler(
            config=config, world=world
        )

    elif config.observation_area_mask == "relative":
        observation_mask_coordinate = LocalObservationCoordinateHandler(
            config=config, world=world
        )
        # observation_mask_coordinate = RelativeObservationCoordinateHandler(
        #     config=config, world=world, observation_space=observation_space
        # )

    else:
        logger.warn(
            f"Unexpected area mask generator is given. config.observation_area_mask: {config.observation_area_mask}"
        )

        raise ValueError()

    return observation_mask_coordinate
