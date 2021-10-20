# -*- coding: utf-8 -*-

"""Source code for observation handler using the merged view method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.worlds import AbstractWorld
from core.worlds.entity import Agent
from omegaconf import DictConfig

from .relative_view_observation import RelativeViewObservaton
from .local_view_observation import LocalViewObservaton
from .abstract_observation import AbstractObservation


class MergedViewObservaton(AbstractObservation):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
        self.visible_range = config.visible_range
        self.visible_radius = config.visible_range // 2

        self.relative_view_observation_handler = RelativeViewObservaton(
            config=config, world=world
        )
        self.local_view_observation_handler = LocalViewObservaton(
            config=config, world=world
        )

    @property
    def observation_space(self):
        return [self.local_view_observation_handler.observation_space[0], *self.relative_view_observation_handler.observation_space[1:]]

    def observation_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        relative_obs = self.relative_view_observation_handler.observation_ind(agents, agent, agent_id)
        local_obs = self.local_view_observation_handler.observation_ind(
            agents, agent, agent_id
        )

        return (relative_obs, local_obs)

    def render(self, state):
        return self.relative_view_observation_handler.render(state[0])
