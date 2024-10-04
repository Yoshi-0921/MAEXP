# -*- coding: utf-8 -*-

"""Source code for abstract observation noise.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig

from core.worlds.abstract_world import AbstractWorld


class AbstractObservationNoise(ABC):
    def __init__(
        self, config: DictConfig, world: AbstractWorld, observation_space: List[int]
    ):
        self.config = config
        self.world = world
        self.observation_space = observation_space
        self.probability_distribution = None

    @abstractmethod
    def add_noise(self, obs, agent, agent_id):
        raise NotImplementedError()
