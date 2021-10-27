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

        for i, a in enumerate(agents):
            diff_x, diff_y = a.xy - agent.xy
            if (
                abs(diff_x) > self.visible_radius
                or abs(diff_y) > self.visible_radius
            ):
                continue

            pos_x, pos_y = self.world.map.coord2ind(
                position=(a.x - agent.x, a.y - agent.y),
                size_x=self.visible_range,
                size_y=self.visible_range,
            )
            if self.config.observation_area_mask == 'relative':
                pos_x += coordinates['obs_x_min']
                pos_y += coordinates['obs_y_min']

            if area_mask[pos_x, pos_y] != -1:
                obs[i, pos_x, pos_y] = 1

        return obs

    def render(self, obs, image, channel):
        # add agent information (Blue)
        for i in range(4):
            image[2] += obs[channel + i]
        # add wandering agent information (Red)
        image[0] += obs[channel + 4]
        image[0] += obs[channel + 5]

        return image, channel + self.get_channel()
