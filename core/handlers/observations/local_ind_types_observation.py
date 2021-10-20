# -*- coding: utf-8 -*-

"""Source code for observation handler using the local view method for different types of agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import numpy as np
import torch

from .local_view_observation import LocalViewObservaton


class LocalIndTypesObservation(LocalViewObservaton):
    @property
    def observation_space(self):
        # 0:agent0, 1:agent1, 2:agent2, 3:agent3, 4:random_agent1, 5:random_agent2, 6:objects, 7:walls
        return [8, self.visible_range, self.visible_range]

    def fill_obs_area(self, obs, agents, agent, agent_id):
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        obs[7, :, :] = self.observation_area_mask[pos_x, pos_y]

        return obs

    def fill_obs_agent(self, obs, agents, agent, agent_id):
        obs[agent_id, self.visible_range // 2, self.visible_range // 2] = 1

        for i, a in enumerate(agents):
            diff_x, diff_y = a.xy - agent.xy
            if abs(diff_x) > 3 or abs(diff_y) > 3 or (diff_x == 0 and diff_y == 0):
                continue

            pos_x, pos_y = self.world.map.coord2ind(
                position=(a.x - agent.x, a.y - agent.y),
                size_x=self.visible_range,
                size_y=self.visible_range,
            )

            if obs[7, pos_x, pos_y] != -1:
                obs[i, pos_x, pos_y] = 1

        return obs

    def fill_obs_object(self, obs, agents, agent, agent_id):
        obs[
            6,
            self.obs_x_min: self.obs_x_max,
            self.obs_y_min: self.obs_y_max
        ] = torch.from_numpy(
            np.where(
                obs[
                    7,
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

    def render(self, state):
        image = torch.zeros((3, *self.observation_space[1:]))
        obs = state.permute(0, 2, 1)

        # add agent information (Blue)
        for i in range(4):
            image[2] += obs[i]
        # add observing agent (Green)
        image[2, self.visible_range // 2, self.visible_range // 2] = 0
        image[1, self.visible_range // 2, self.visible_range // 2] = 1
        # add wandering agent information (Red)
        image[0] += obs[4]
        image[0] += obs[5]
        # add object information (Yellow)
        image[torch.tensor([0, 1])] += obs[6]
        # add invisible area information (White)
        image -= (obs[7])

        image = image.clamp(min=0, max=1)

        return image