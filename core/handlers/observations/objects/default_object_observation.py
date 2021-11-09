# -*- coding: utf-8 -*-

"""Source code for default observation handler for objects.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import numpy as np
import torch
from core.utils.color import RGB_COLORS

from ..abstract_observation_handler import AbstractObservationHandler


class DefaultObjectObservationHandler(AbstractObservationHandler):
    def get_channel(self):
        return 1

    def fill(self, agents, agent, agent_id, area_mask, coordinates):
        obs = torch.zeros(1, *area_mask.shape)

        obs[
            0,
            coordinates["obs_x_min"]: coordinates["obs_x_max"],
            coordinates["obs_y_min"]: coordinates["obs_y_max"],
        ] = torch.from_numpy(
            area_mask[
                coordinates["area_x_min"]: coordinates["area_x_max"],
                coordinates["area_y_min"]: coordinates["area_y_max"],
            ]
            * self.world.map.objects_matrix[
                :,
                coordinates["map_x_min"]: coordinates["map_x_max"],
                coordinates["map_y_min"]: coordinates["map_y_max"],
            ].sum(axis=0)
        )

        return obs

    def render(self, obs, image, channel):
        # add object information (Yellow)
        rgb = RGB_COLORS["yellow"]
        rgb = np.expand_dims(np.asarray(rgb), axis=(1, 2))
        image += obs[channel] * rgb

        return image, channel + self.get_channel()
