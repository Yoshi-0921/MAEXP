# -*- coding: utf-8 -*-

"""Source code for multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC
from typing import List

import numpy as np
from core.maps.abstract_map import AbstractMap
from .entity import Agent, Object
from omegaconf import DictConfig


class AbstractWorld(ABC):
    def __init__(self, config: DictConfig, world_map: AbstractMap):
        self.config = config
        self.map = world_map
        self.agents: List[Agent] = []
        self.objects: List[Object] = []

    @property
    def entities(self):
        return self.agents + self.objects

    def reset_all(self):
        self.reset_agents()
        self.reset_objects()
        self.reset_map()

    def reset_agents(self):
        self.agents = []

    def reset_objects(self):
        self.objects = []

    def reset_map(self):
        self.map.reset_all()

    def step(self):
        force = [None] * len(self.agents)
        force = self.apply_action_force(force)
        force = self.apply_environment_force(force)
        self.integrate_state(force)

    def apply_action_force(self, force):
        for agent_id, agent in enumerate(self.agents):
            force[agent_id] = agent.action
        return force

    def apply_environment_force(self, force):
        for agent_id, agent in enumerate(self.agents):

            # Check if there is a collision against other agents.
            next_pos = agent.pos + force[agent_id]
            for i, a in enumerate(self.agents):
                if agent_id == i:
                    continue

                if all(next_pos == a.pos):
                    force[agent_id] = np.zeros(2, dtype=np.int8)
                    agent.collide_agents = True
                    break

            # Check if there is a collision against wall.
            next_pos = self.map.coord2ind(next_pos)
            if self.map.wall_matrix[next_pos] == 1:
                force[agent_id] = np.zeros(2, dtype=np.int8)
                agent.collide_walls = True

        return force

    def integrate_state(self, force):
        self.map.reset_agents()
        for agent_id, agent in enumerate(self.agents):
            agent.push(force[agent_id])
            self.map.agents_matrix(self.map.coord2ind(agent.xy))
