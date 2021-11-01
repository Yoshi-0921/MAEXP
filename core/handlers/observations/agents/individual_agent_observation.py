# -*- coding: utf-8 -*-

"""Source code for observation handler for individual agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import torch

from ..abstract_observation_handler import AbstractObservationHandler


class IndividualAgentObservationHandler(AbstractObservationHandler):
    def get_channel(self):
        # 0:agent0, 1:agent1, 2:agent2, 3:agent3, 4:random_agent1, 5:random_agent2
        return 6

    def fill(self, agents, agent, agent_id, area_mask, coordinates):
        obs = torch.zeros(6, *area_mask.shape)

        for obs_i, agent_matrix in zip(obs, self.world.map.agents_matrix):
            obs_i[
                coordinates["obs_x_min"]: coordinates["obs_x_max"],
                coordinates["obs_y_min"]: coordinates["obs_y_max"],
            ] = torch.from_numpy(
                area_mask[
                    coordinates["area_x_min"]: coordinates["area_x_max"],
                    coordinates["area_y_min"]: coordinates["area_y_max"],
                ]
                * agent_matrix[
                    coordinates["map_x_min"]: coordinates["map_x_max"],
                    coordinates["map_y_min"]: coordinates["map_y_max"],
                ]
            )

        return obs

    def render(self, obs, image, channel):
        # add agent information (Blue)
        for i in range(6):
            image[2] += obs[channel + i]

        return image, channel + self.get_channel()
