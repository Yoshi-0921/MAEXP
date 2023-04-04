"""Source code for default multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import numpy as np
from core.worlds import AbstractWorld
from omegaconf import DictConfig
from core.handlers.observations.observation_handler import ObservationHandler

from .default_environment import DefaultEnvironment


class ObservationStatsEnvironment(DefaultEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        self.view_method = config.observation_area_mask
        self.map_SIZE_X, self.map_SIZE_Y = config.map.SIZE_X, config.map.SIZE_Y
        self.observation_stats = []
        for _ in range(len(self.agents)):
            self.observation_stats.append(
                np.zeros(
                    shape=self.observation_handler.observation_space,
                    dtype=np.float32,
                )
            )

    def step(self, action_n: List[np.array], order: List[int]):
        reward_n: List[np.array] = []
        done_n: List[np.array] = []
        obs_n: List[np.array] = []

        for agent_id, agent in enumerate(self.agents):
            self.action_ind(action_n[agent_id], agent)

        # excecute action in the environment
        self.world.step(order)

        # obtain the outcome from the environment for each agent
        for agent_id, agent in enumerate(self.agents):
            reward_n.append(self.reward_ind(self.agents, agent, agent_id))
            done_n.append(self.done_ind(self.agents, agent, agent_id))
            observation_ind = self.observation_ind(self.agents, agent, agent_id)
            if self.view_method == "relative":
                relative_x = ObservationHandler.decode_relative_state(
                    state=observation_ind, observation_size=[self.map_SIZE_X, self.map_SIZE_Y]
                )
                self.observation_stats[agent_id] += relative_x.detach().numpy()
            elif self.view_method == "local":
                self.observation_stats[agent_id] += observation_ind["local"].detach().numpy()
            obs_n.append(observation_ind)

        self.observation_handler.step(self.agents)

        self.current_step += 1
        self.heatmap_objects_left += self.world.map.objects_matrix

        return reward_n, done_n, obs_n
