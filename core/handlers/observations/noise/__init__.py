# -*- coding: utf-8 -*-

"""Builds observation noise handler.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig

from .abstract_observation_noise import AbstractObservationNoise
from .dist_observation_noise import (DistObservatonNoise,
                                     ThresholdDistObservationNoise)
from .flat_observation_noise import FlatObservatonNoise
from .non_observation_noise import NonObservationNoise
from .flip_observation_noise import FlipObservatonNoise

logger = initialize_logging(__name__)

__all__ = [
    "AbstractObservationNoise",
    "NonObservationNoise",
    "DistObservatonNoise",
    "FlatObservatonNoise",
    "ThresholdDistObservationNoise",
    "FlipObservatonNoise"
]


def generate_observation_noise(config: DictConfig, world: AbstractWorld, observation_space: List[int]) -> AbstractObservationNoise:
    if not config.observation_noise:
        return NonObservationNoise(config=config, world=world, observation_space=observation_space)

    elif config.observation_noise == "sensing_dist":
        noise = DistObservatonNoise(config=config, world=world, observation_space=observation_space)

    elif config.observation_noise == "threshold_sensing_dist":
        noise = ThresholdDistObservationNoise(config=config, world=world, observation_space=observation_space)

    elif config.observation_noise == "flip":
        noise = FlipObservatonNoise(config=config, world=world, observation_space=observation_space)

    elif config.observation_noise == 'flat':
        noise = FlatObservatonNoise(config=config, world=world, observation_space=observation_space)

    else:
        logger.warn(
            f"Unexpected view_method is given. config.view_method: {config.view_method}"
        )

        raise ValueError()

    return noise
