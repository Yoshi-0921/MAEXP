# -*- coding: utf-8 -*-

"""Source code for observation handler using the local simple view method.
This observation method supposes that agents can observe objects behind walls.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch
from core.worlds.entity import Agent

from .abstract_observation import AbstractObservation


class LocalSimpleObservation(AbstractObservation):
    @property
    def observation_space(self):
        # 0:agents, 1:objects, 2:walls
        return [3, self.visible_range, self.visible_range]

    def get_mask_coordinates(self, agent):
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        self.obs_x_min = abs(min(0, pos_x - self.visible_range // 2))
        self.obs_x_max = self.visible_range - max(0, pos_x + self.visible_range // 2 - (self.world.map.SIZE_X - 1))
        self.obs_y_min = abs(min(0, pos_y - self.visible_range // 2))
        self.obs_y_max = self.visible_range - max(0, pos_y + self.visible_range // 2 - (self.world.map.SIZE_Y - 1))
        self.global_x_min = max(0, pos_x - self.visible_range // 2)
        self.global_x_max = min(
            self.world.map.SIZE_X - 1, pos_x + self.visible_range // 2
        )
        self.global_y_min = max(0, pos_y - self.visible_range // 2)
        self.global_y_max = min(
            self.world.map.SIZE_Y - 1, pos_y + self.visible_range // 2
        )

    def observation_ind(self, agent: Agent, agent_id: int):
        obs = torch.zeros(self.observation_space)
        offset = 0

        self.get_mask_coordinates(agent)

        # input walls
        obs = self.fill_obs_area(obs, agent, agent_id, offset, offset)

        # input objects within sight
        obs = self.fill_obs_object(obs, agent, agent_id, offset, offset)

        # input agents within sight
        obs = self.fill_obs_agent(obs, agent, agent_id, offset, offset)

        return obs

    def fill_obs_area(self, obs, agent, agent_id, offset_x, offset_y):
        obs[2] -= 1
        obs[
            2,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max,
        ] *= torch.from_numpy(self.world.map.wall_matrix[
            self.global_x_min: (self.global_x_max + 1),
            self.global_y_min: (self.global_y_max + 1),
        ])

        return obs

    def fill_obs_agent(self, obs, agent, agent_id, offset_x, offset_y):
        obs[
            0,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max,
        ] += torch.from_numpy(self.world.map.agents_matrix[
            self.global_x_min: (self.global_x_max + 1),
            self.global_y_min: (self.global_y_max + 1),
        ])

        return obs

    def fill_obs_object(self, obs, agent, agent_id, offset_x, offset_y):
        obs[
            1,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max,
        ] += torch.from_numpy(self.world.map.objects_matrix[
            self.global_x_min: (self.global_x_max + 1),
            self.global_y_min: (self.global_y_max + 1),
        ])

        return obs

    def render(self, state):
        image = torch.zeros((self.observation_space))
        obs = state.permute(0, 2, 1)

        # add agent information (Blue)
        image[2] += obs[0]
        # add object information (Yellow)
        image[torch.tensor([0, 1])] += obs[1]
        # add invisible area information (White)
        image -= obs[2]

        image = image.clamp(min=0, max=1)

        return image
