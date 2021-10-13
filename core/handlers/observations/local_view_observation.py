# -*- coding: utf-8 -*-

"""Source code for observation handler using the local view method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import numpy as np
import torch
from core.worlds.entity import Agent

from .abstract_observation import AbstractObservation


class LocalViewObservaton(AbstractObservation):
    @property
    def observation_space(self) -> List[int]:
        # 0:agents, 1:objects, 2:visible area
        return [3, self.visible_range, self.visible_range]

    def get_mask_coordinates(self, agent):
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        self.obs_x_min = abs(min(0, pos_x - self.visible_radius))
        self.obs_x_max = self.visible_range - max(
            0, pos_x + self.visible_radius - (self.world.map.SIZE_X - 1)
        )
        self.obs_y_min = abs(min(0, pos_y - self.visible_radius))
        self.obs_y_max = self.visible_range - max(
            0, pos_y + self.visible_radius - (self.world.map.SIZE_Y - 1)
        )
        self.global_x_min = max(0, pos_x - self.visible_radius)
        self.global_x_max = min(self.world.map.SIZE_X - 1, pos_x + self.visible_radius)
        self.global_y_min = max(0, pos_y - self.visible_radius)
        self.global_y_max = min(self.world.map.SIZE_Y - 1, pos_y + self.visible_radius)

    def observation_ind(self, agents: List[Agent], agent: Agent, agent_id: int) -> torch.Tensor:
        obs = torch.zeros(self.observation_space)

        self.get_mask_coordinates(agent)

        # input walls and invisible area
        obs = self.fill_obs_area(obs, agents, agent, agent_id)

        # input objects within sight
        obs = self.fill_obs_object(obs, agents, agent, agent_id)

        # input agents within sight
        obs = self.fill_obs_agent(obs, agents, agent, agent_id)

        # add observation noise
        obs = self.fill_obs_noise(obs, agents, agent, agent_id)

        return obs

    def fill_obs_area(self, obs, agents, agent, agent_id) -> torch.Tensor:
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        obs[2, :, :] = self.observation_area_mask[pos_x, pos_y]

        return obs

    def fill_obs_agent(self, obs, agents, agent, agent_id) -> torch.Tensor:
        obs[
            0,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max
        ] = torch.from_numpy(
            np.where(
                obs[
                    2,
                    self.obs_x_min: self.obs_x_max,
                    self.obs_y_min: self.obs_y_max
                ] != -1,
                self.world.map.agents_matrix[
                    self.global_x_min: (self.global_x_max + 1),
                    self.global_y_min: (self.global_y_max + 1),
                ],
                0
            )
        )

        return obs

    def fill_obs_object(self, obs, agents, agent, agent_id) -> torch.Tensor:
        obs[
            1,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max
        ] = torch.from_numpy(
            np.where(
                obs[
                    2,
                    self.obs_x_min: self.obs_x_max,
                    self.obs_y_min: self.obs_y_max
                ] != -1,
                self.world.map.objects_matrix[
                    self.global_x_min: (self.global_x_max + 1),
                    self.global_y_min: (self.global_y_max + 1),
                ],
                0
            )
        )

        return obs

    def fill_obs_noise(self, obs, agents, agent, agent_id) -> torch.Tensor:
        return self.observation_noise.add_noise(
            obs, agent, agent_id
        )

    def render(self, state) -> torch.Tensor:
        image = torch.zeros(self.observation_space)
        obs = state.permute(0, 2, 1)

        # add agent information (Blue)
        image[2] += obs[0]
        # add observing agent (Green)
        image[2, self.visible_range // 2, self.visible_range // 2] = 0
        image[1, self.visible_range // 2, self.visible_range // 2] = 1
        # add object information (Yellow)
        image[torch.tensor([0, 1])] += obs[1]
        # add invisible area information (White)
        image -= obs[2]

        return image
