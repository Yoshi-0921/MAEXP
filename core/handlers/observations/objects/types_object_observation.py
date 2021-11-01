# -*- coding: utf-8 -*-

"""Source code for types observation handler for obects.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch
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
        # add object information (Yellow)
        for i, (red, green) in zip(range(self.config.type_objects), [(1., 1.), (1., 0.5), (0.5, 1.)]):
            image[0] += obs[channel + i] * red
            image[1] += obs[channel + i] * green

        return image, channel + self.get_channel()
