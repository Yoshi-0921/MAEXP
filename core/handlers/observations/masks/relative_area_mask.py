# -*- coding: utf-8 -*-

"""Returns relative observation mask.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import numpy as np
import torch
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig
from tqdm import tqdm
from .local_area_mask import generate_each_mask
from .abstract_observation_coordinate_handler import (
    AbstractObservationCoordinateHandler,
)


class RelativeObservationCoordinateHandler(AbstractObservationCoordinateHandler):
    def get_mask_coordinates(self, agent):
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        obs_x_min = max(0, pos_x - self.visible_radius)
        obs_x_max = min(self.world.map.SIZE_X - 1, pos_x + self.visible_radius) + 1
        obs_y_min = max(0, pos_y - self.visible_radius)
        obs_y_max = min(self.world.map.SIZE_Y - 1, pos_y + self.visible_radius) + 1

        return {
            "obs_x_min": obs_x_min,
            "obs_x_max": obs_x_max,
            "obs_y_min": obs_y_min,
            "obs_y_max": obs_y_max,
            "area_x_min": obs_x_min,
            "area_x_max": obs_x_max,
            "area_y_min": obs_y_min,
            "area_y_max": obs_y_max,
            "map_x_min": obs_x_min,
            "map_x_max": obs_x_max,
            "map_y_min": obs_y_min,
            "map_y_max": obs_y_max,
        }


def generate_relative_area_mask(
    config: DictConfig, world: AbstractWorld
) -> torch.Tensor:
    mask = (
        np.zeros(
            (
                world.map.SIZE_X,
                world.map.SIZE_Y,
                world.map.SIZE_X,
                world.map.SIZE_Y,
            ),
            dtype=np.int8,
        )
        - 1
    )
    visible_range = config.visible_range
    visible_radius = config.visible_range // 2
    with tqdm(total=world.map.SIZE_X - 2) as pbar:
        pbar.set_description("Generating mask")
        for x in range(1, world.map.SIZE_X - 1):
            for y in range(1, world.map.SIZE_Y - 1):
                global_x_min = abs(min(0, x - visible_radius))
                global_x_max = visible_range - max(
                    0, x + visible_radius - (world.map.SIZE_X - 1)
                )
                global_y_min = abs(min(0, y - visible_radius))
                global_y_max = visible_range - max(
                    0, y + visible_radius - (world.map.SIZE_Y - 1)
                )
                obs_x_min = max(0, x - visible_radius)
                obs_x_max = min(world.map.SIZE_X - 1, x + visible_radius) + 1
                obs_y_min = max(0, y - visible_radius)
                obs_y_max = min(world.map.SIZE_Y - 1, y + visible_radius) + 1

                if config.transparent_observation:
                    mask[
                        x, y, obs_x_min:obs_x_max, obs_y_min:obs_y_max
                    ] = world.map.wall_matrix[obs_x_min:obs_x_max, obs_y_min:obs_y_max] * (-1)

                else:
                    mask[
                        x, y, obs_x_min:obs_x_max, obs_y_min:obs_y_max
                    ] = generate_each_mask(x, y, world.map, config.visible_range)[
                        global_x_min:global_x_max, global_y_min:global_y_max
                    ]
            pbar.update(1)
        pbar.close()

    return mask + 1
