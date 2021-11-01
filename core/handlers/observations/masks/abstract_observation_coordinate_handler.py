# -*- coding: utf-8 -*-

"""Source code for abstract observation coordinate handler.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig


class AbstractObservationCoordinateHandler(ABC):
    def __init__(
        self, config: DictConfig, world: AbstractWorld, observation_space: List[int]
    ):
        self.config = config
        self.world = world
        self.observation_space = observation_space
        self.visible_range = config.visible_range
        self.visible_radius = config.visible_range // 2

    @abstractmethod
    def get_mask_coordinates(self, agent) -> Dict[str, int]:
        raise NotImplementedError()
