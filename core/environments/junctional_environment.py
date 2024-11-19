"""Source code for sequential-tasks multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import random
from typing import List

import numpy as np
from omegaconf import DictConfig

from core.handlers.observations import generate_observation_handler
from core.worlds import AbstractWorld
from core.worlds.entity import Agent

from .abstract_environment import AbstractEnvironment
from .default_environment import DefaultEnvironment


class JunctionalEnvironment(DefaultEnvironment):
    def generate_objects(self, num_objects: int = None, object_type: int = None):
        num_objects = num_objects or self.config.num_objects
        self._generate_objects(num_objects)

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0
        if self.agent_tasks[agent_id] == "-1":
            return reward

        for object_type in self.agent_tasks[agent_id]:
            if (
                self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y]
                == self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y]
                == 1
            ):
                reward = 1.0
                self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y] = 0
                self.objects_completed += 1
                self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1

                if int(object_type) == 0:
                    if self.config.keep_objects_num:
                        self.generate_objects(1, 0)

                if int(object_type) == 1:
                    if agent_id in [2, 3]:
                        self.generate_object_at(3, a_pos_x, a_pos_y)
                    if agent_id in [4, 5]:
                        self.generate_object_at(2, a_pos_x, a_pos_y)
                elif int(object_type) == 2:
                    self.generate_object_at(4, a_pos_x, a_pos_y)
                elif int(object_type) < self.config.type_objects - 1:
                    self.generate_object_at(int(object_type)+1, a_pos_x, a_pos_y)
                elif int(object_type) == self.config.type_objects - 1:
                    if self.config.keep_objects_num:
                        self.generate_objects(1, 0)

        # negative reward for collision with other agents
        if agent.collide_agents:
            reward = -1.0
            agent.collide_agents = False
            self.heatmap_agents_collision[a_pos_x, a_pos_y] += 1
            self.agents_collided += 1

        # negative reward for collision against walls
        if agent.collide_walls:
            reward = -1.0
            agent.collide_walls = False
            self.heatmap_wall_collision[a_pos_x, a_pos_y] += 1
            self.walls_collided += 1

        return reward
