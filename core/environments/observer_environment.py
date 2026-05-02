"""Source code for observer-tasks multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import numpy as np
from omegaconf import DictConfig

from core.worlds import AbstractWorld
from core.worlds.entity import Agent

from .default_environment import DefaultEnvironment


class ObserverEnvironment(DefaultEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        self.observed = {i:False for i in range(len(self.agents))}

    def generate_objects(self, num_objects: int = None, object_type: int = None):
        num_objects = num_objects or self.config.num_objects
        self._generate_objects(num_objects)

    def reset(self):
        obs_n = super().reset()
        self.observed = {i:False for i in range(len(self.agents))}
        return obs_n

    def step(self, action_n: List[np.array], order: List[int]):
        reward_n, done_n, obs_n = super().step(action_n=action_n, order=order)
        self.observed = {i:False for i in range(len(self.agents))}
        return reward_n, done_n, obs_n

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        agent_x, agent_y = agent.xy
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0
        if self.agent_tasks[agent_id] == "-1":
            return reward

        if self.agent_tasks[agent_id] == "o":
            if(
                self.world.map.objects_matrix[0, a_pos_x, a_pos_y]
                == self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y]
                == 1
            ):
                reward = 0.1
                self.world.map.objects_matrix[0, a_pos_x, a_pos_y] = 0
                self.objects_completed += 1
                self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1
                if self.config.keep_objects_num:
                    self.generate_objects(1, 0)

            if self.observed[agent_id]:
                reward = 1.0

        else:
            for object_type in self.agent_tasks[agent_id]:
                if (
                    self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y]
                    == self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y]
                    == 1
                ):
                    reward = 0.1

                    for observer_id, observer in enumerate(agents[-3:]):
                        observer_x, observer_y = observer.xy
                        if (observer_x - 3 <= agent_x <= observer_x + 3) and (observer_y - 3 <= agent_y <= observer_y + 3):
                            self.observed[len(self.agents)-3+observer_id] = True
                            reward = 1.0

                    self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y] = 0
                    self.objects_completed += 1
                    self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1
                    if self.config.keep_objects_num:
                        self.generate_objects(1, int(object_type))

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
