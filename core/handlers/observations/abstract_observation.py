# -*- coding: utf-8 -*-

"""Source code for abstract multi-agent observation method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod

from core.worlds import AbstractWorld
from omegaconf import DictConfig
from core.worlds.entity import Agent


class AbstractObservation(ABC):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
        self.visible_range = config.visible_range

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @abstractmethod
    def observation_ind(self, agent: Agent):
        raise NotImplementedError()

    @abstractmethod
    def fill_obs_area(self, obs, agent, offset_x, offset_y):
        raise NotImplementedError()

    @abstractmethod
    def fill_obs_agent(self, obs, agent, offset_x, offset_y):
        raise NotImplementedError()

    @abstractmethod
    def fill_obs_object(self, obs, agent, offset_x, offset_y):
        raise NotImplementedError()

    @abstractmethod
    def render(self, state):
        raise NotImplementedError()

    def reset(self, agents):
        obs_n = []
        for agent in agents:
            obs_n.append(self.observation_ind(agent))

        return obs_n

    def step(self):
        pass
