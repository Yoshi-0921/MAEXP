# -*- coding: utf-8 -*-

"""Builds observation handler used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from omegaconf import DictConfig

from core.worlds.abstract_world import AbstractWorld

from .merged_observtion_handler import MergedObservationHandler
from .observation_handler import ObservationHandler

__all__ = [
    "ObservationHandler",
    "MergedObservationHandler"
]


def generate_observation_handler(config: DictConfig, world: AbstractWorld):
    if config.observation_area_mask == 'merged':
        obs = MergedObservationHandler(config=config, world=world)

    else:
        obs = ObservationHandler(config=config, world=world)

    return obs
