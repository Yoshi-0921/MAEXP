# -*- coding: utf-8 -*-

"""Source code for abstract multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from core.worlds.abstract_world import AbstractWorld
from core.worlds.entity import Agent
from omegaconf import DictConfig


class AbstractEnvironment(ABC):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
        self.agents = self.world.agents
        self.num_agents = len(self.agents)

    def render_world(self):
        self.world.render()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def observation(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action_n: List[np.array]):
        raise NotImplementedError()

    @abstractmethod
    def reward(self):
        raise NotImplementedError()

    @abstractmethod
    def action_ind(self, action: int, agent: Agent):
        raise NotImplementedError()

    @abstractmethod
    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        raise NotImplementedError()

    @abstractmethod
    def done_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        raise NotImplementedError()

    @abstractmethod
    def observation_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        raise NotImplementedError()
