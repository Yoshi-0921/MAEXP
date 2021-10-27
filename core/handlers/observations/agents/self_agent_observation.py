# -*- coding: utf-8 -*-

"""Source code for self observation handler for agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch
from ..abstract_observation_handler import AbstractObservationHandler


class SelfAgentObservationHandler(AbstractObservationHandler):
    def get_channel(self):
        # 0: itself, 1: other agents
        return 2

    def fill(self, agents, agent, agent_id, area_mask, coordinates):
        obs = torch.zeros(2, *area_mask.shape)
        obs[
            1,
            coordinates["obs_x_min"]: coordinates["obs_x_max"],
            coordinates["obs_y_min"]: coordinates["obs_y_max"],
        ] = torch.from_numpy(
            area_mask[
                coordinates["area_x_min"]: coordinates["area_x_max"],
                coordinates["area_y_min"]: coordinates["area_y_max"],
            ]
            * self.world.map.agents_matrix[
                coordinates["map_x_min"]: coordinates["map_x_max"],
                coordinates["map_y_min"]: coordinates["map_y_max"],
            ]
        )

        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        obs[0, pos_x, pos_y] = 1
        obs[1, pos_x, pos_y] = 0

        return obs

    def render(self, obs, image, channel):
        # add agent information (Blue)
        image[2] += obs[channel]
        image[2] += obs[channel + 1]

        return image, channel + self.get_channel()
