"""Source code for default test multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.environments.default_environment import DefaultEnvironment
from core.worlds.entity import Agent
import numpy as np

class TestEnvironment(DefaultEnvironment):
    def __init__(self, config):
        super().__init__(config=config)
        self.acc_objects_completed_individually = np.zeros(
            shape=(self.num_agents),
            dtype=np.int32,
        )
        self.acc_objects_completed_correctly_type = np.zeros(
            shape=(self.num_agents),
            dtype=np.int32,
        )
        self.acc_objects_completed_correctly_area = np.zeros(
            shape=(self.num_agents),
            dtype=np.int32,
        )

    def reset(self):
        obs_n = super().reset()
        self.objects_completed_successfully = 0
        self.objects_completed_individually = np.zeros(
            shape=(self.num_agents),
            dtype=np.int32,
        )
        self.objects_completed_correctly_type = np.zeros(
            shape=(self.num_agents),
            dtype=np.int32,
        )
        self.objects_completed_correctly_area = np.zeros(
            shape=(self.num_agents),
            dtype=np.int32,
        )
        return obs_n

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0

        if hasattr(self.world.map, 'specified_object_types'):
            agent_tasks = self.world.map.specified_object_types[agent_id]
        else:
            agent_tasks = self.agent_tasks[agent_id]

        if agent_tasks == "-1":
            return reward

        for object_type in range(self.type_objects):
            if (
                self.world.map.objects_matrix[object_type, a_pos_x, a_pos_y]
                == 1
            ):
                if (
                    self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y] == 1
                ):
                    self.objects_completed_correctly_area[agent_id] += 1
                    if str(object_type) in agent_tasks:
                        reward = 1.0
                        self.objects_completed_successfully += 1
                        self.objects_completed_correctly_type[agent_id] += 1
                self.world.map.objects_matrix[object_type, a_pos_x, a_pos_y] = 0
                self.objects_completed += 1
                self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1
                self.objects_completed_individually[agent_id] += 1
                if self.config.keep_objects_num:
                    self.generate_objects(1, object_type)

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

    def accumulate_heatmap(self):
        super().accumulate_heatmap()
        self.acc_objects_completed_individually += self.objects_completed_individually
        self.acc_objects_completed_correctly_type += self.objects_completed_correctly_type
        self.acc_objects_completed_correctly_area += self.objects_completed_correctly_area
