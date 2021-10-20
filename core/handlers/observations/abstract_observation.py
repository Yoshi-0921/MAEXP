# -*- coding: utf-8 -*-

"""Source code for abstract multi-agent observation method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from typing import List

import torch
from core.worlds import AbstractWorld
from core.worlds.entity import Agent
from omegaconf import DictConfig

from .masks import generate_observation_area_mask
from .noises import generate_observation_noise


class AbstractObservation(ABC):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
        self.visible_range = config.visible_range
        self.visible_radius = config.visible_range // 2
        self.observation_noise = generate_observation_noise(
            config=config, world=world, observation_space=self.observation_space
        )
        self.observation_area_mask = generate_observation_area_mask(
            config=config, world=world
        )

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @abstractmethod
    def observation_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        raise NotImplementedError()

    def fill_obs_area(self, obs, agent, agent_id):
        raise NotImplementedError()

    def fill_obs_agent(self, obs, agent, agent_id):
        raise NotImplementedError()

    def fill_obs_object(self, obs, agent, agent_id):
        raise NotImplementedError()

    def fill_obs_noise(self, obs, agent, agent_id):
        raise NotImplementedError()

    @abstractmethod
    def render(self, state):
        raise NotImplementedError()

    def reset(self, agents) -> torch.Tensor:
        obs_n = []
        for agent_id, agent in enumerate(agents):
            obs_n.append(self.observation_ind(agents, agent, agent_id))

        return obs_n

    def step(self, agents):
        pass
