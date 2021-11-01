# -*- coding: utf-8 -*-

"""Source code for simple observation handler for objects.
This observation method supposes that agents can observe objects behind walls.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch
from ..abstract_observation_handler import AbstractObservationHandler


class SimpleObjectObservationHandler(AbstractObservationHandler):
    def get_channel(self):
        return 1

    def fill(self, agents, agent, agent_id, area_mask, coordinates):
        obs = torch.zeros(1, *area_mask.shape)

        obs[
            0,
            coordinates["obs_x_min"]: coordinates["obs_x_max"],
            coordinates["obs_y_min"]: coordinates["obs_y_max"],
        ] = torch.from_numpy(
            self.world.map.objects_matrix[
                :,
                coordinates["map_x_min"]: coordinates["map_x_max"],
                coordinates["map_y_min"]: coordinates["map_y_max"],
            ].sum(axis=0)
        )

        return obs

    def render(self, obs, image, channel):
        # add object information (Yellow)
        image[torch.tensor([0, 1])] += obs[channel]

        return image, channel + self.get_channel()
