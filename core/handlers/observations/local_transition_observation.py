# -*- coding: utf-8 -*-

"""Source code for observation handler using the local transition view method.
This observation method supposes that agents can observe objects behind walls.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import numpy as np
import torch
from core.worlds import AbstractWorld
from omegaconf import DictConfig

from .local_simple_observation import LocalSimpleObservation


class LocalTransitionObservation(LocalSimpleObservation):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        self.num_agents = config.num_agents
        self.past_step = config.past_step
        assert 0 < self.config.past_step

    def reset(self, agents):
        self.past_global_agents = [self.get_past_global_agents(agents) for _ in range(self.past_step)]
        return super().reset(agents)

    def get_past_global_agents(self, agents):
        global_agents = [self.world.map.agents_matrix.copy() for _ in range(self.num_agents)]
        for agent_id, agent in enumerate(agents):
            self.get_mask_coordinates(agent)
            mask = np.zeros_like(self.world.map.agents_matrix)
            mask[self.global_x_min: (self.global_x_max + 1), self.global_y_min: (self.global_y_max + 1)] += 1
            global_agents[agent_id] *= mask

        return global_agents

    def step(self, agents):
        # past_global_agents: [[t-1], [t-2], ..., [t-n]]
        self.past_global_agents.pop()
        self.past_global_agents.insert(0, self.get_past_global_agents(agents))

    @property
    def observation_space(self):
        # 0:agents, 1:agents(t-1), 2:agents(t-2), ..., -2:objects, -1:walls
        return [3 + self.past_step, self.visible_range, self.visible_range]

    def fill_obs_area(self, obs, agent, agent_id, offset_x, offset_y):
        obs[-1] -= 1
        obs[
            -1,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max,
        ] *= torch.from_numpy(self.world.map.wall_matrix[
            self.global_x_min: (self.global_x_max + 1),
            self.global_y_min: (self.global_y_max + 1),
        ])

        return obs

    def fill_obs_agent(self, obs, agent, agent_id, offset_x, offset_y):
        obs = super().fill_obs_agent(obs, agent, agent_id, offset_x, offset_y)

        for t in range(self.past_step):
            obs[
                t + 1,
                self.obs_x_min: self.obs_x_max,
                self.obs_y_min: self.obs_y_max,
            ] += torch.from_numpy(self.past_global_agents[t][agent_id][
                self.global_x_min: (self.global_x_max + 1),
                self.global_y_min: (self.global_y_max + 1),
            ])

        return obs

    def fill_obs_object(self, obs, agent, agent_id, offset_x, offset_y):
        obs[
            -2,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max,
        ] += torch.from_numpy(self.world.map.objects_matrix[
            self.global_x_min: (self.global_x_max + 1),
            self.global_y_min: (self.global_y_max + 1),
        ])

        return obs

    def render(self, state):
        image = torch.zeros((3, *self.observation_space[1:]))
        obs = state.permute(0, 2, 1)

        # add agent information (Blue)
        image[2] += obs[0]
        for t in range(1, self.past_step + 1):
            image[2] += (obs[t] * (0.5 ** t))
        # add object information (Yellow)
        image[torch.tensor([0, 1])] += obs[-2]
        # add invisible area information (White)
        image -= obs[-1]

        image = image.clamp(min=0, max=1)

        return image
