# -*- coding: utf-8 -*-

"""Source code for observation handler using the merged view method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from core.worlds import AbstractWorld
from core.worlds.entity import Agent
from omegaconf import DictConfig

from .observation_handler import ObservationHandler


class MergedObservationHandler:
    def __init__(self, config: DictConfig, world: AbstractWorld):
        config.observation_area_mask = 'relative'
        self.relative_view_observation_handler = ObservationHandler(
            config=config, world=world
        )
        config.observation_area_mask = 'local'
        self.local_view_observation_handler = ObservationHandler(
            config=config, world=world
        )

    @property
    def observation_space(self):
        return [self.local_view_observation_handler.observation_space[0], *self.relative_view_observation_handler.observation_space[1:]]

    def observation_ind(self, agents: List[Agent], agent: Agent, agent_id: int) -> torch.Tensor:
        relative_obs = self.relative_view_observation_handler.observation_ind(agents, agent, agent_id)
        local_obs = self.local_view_observation_handler.observation_ind(
            agents, agent, agent_id
        )

        return (relative_obs, local_obs)

    def render(self, state: torch.Tensor) -> torch.Tensor:
        return self.relative_view_observation_handler.render(state[0])

    def reset(self, agents: List[Agent]) -> torch.Tensor:
        relative_obs_n = self.relative_view_observation_handler.reset(agents)
        local_obs_n = self.local_view_observation_handler.reset(agents)

        obs_n = [(relative_obs, local_obs) for relative_obs, local_obs in zip(relative_obs_n, local_obs_n)]

        return obs_n

    def step(self, agents: List[Agent]):
        self.relative_view_observation_handler.step(agents)
        self.local_view_observation_handler.step(agents)
