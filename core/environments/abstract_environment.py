# -*- coding: utf-8 -*-

"""Source code for abstract multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from core.worlds.abstract_world import AbstractWorld


class AbstractEnvironment(ABC):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
        self.agents = self.world.agents
        self.objects = self.world.objects
        self.num_agents = len(self.agents)
        self.num_objects = len(self.objects)

    @abstractmethod
    def generate_world(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def observation(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    @abstractmethod
    def reward(self):
        raise NotImplementedError()
