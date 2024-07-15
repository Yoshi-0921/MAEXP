# -*- coding: utf-8 -*-

"""Source code for types observation handler for obects.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import numpy as np
import torch

from core.utils.color import RGB_COLORS

from ..abstract_observation_handler import AbstractObservationHandler


class TypesObjectObservationHandler(AbstractObservationHandler):
    def get_channel(self):
        return self.config.type_objects

    def fill(self, agents, agent, agent_id, area_mask, coordinates):
        obs = torch.zeros(self.get_channel(), *area_mask.shape)

        for obs_i, object_matrix in zip(obs, self.world.map.objects_matrix):
            obs_i[
                coordinates["obs_x_min"]: coordinates["obs_x_max"],
                coordinates["obs_y_min"]: coordinates["obs_y_max"],
            ] = torch.from_numpy(
                area_mask[
                    coordinates["area_x_min"]: coordinates["area_x_max"],
                    coordinates["area_y_min"]: coordinates["area_y_max"],
                ]
                * object_matrix[
                    coordinates["map_x_min"]: coordinates["map_x_max"],
                    coordinates["map_y_min"]: coordinates["map_y_max"],
                ]
            )

        return obs

    def render(self, obs, image, channel):
        for i, color in enumerate(self.objects_color):
            rgb = RGB_COLORS[color]
            rgb = np.expand_dims(np.asarray(rgb), axis=(1, 2))
            image += obs[channel + i] * rgb

        return image, channel + self.get_channel()
