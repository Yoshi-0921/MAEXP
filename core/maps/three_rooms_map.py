# -*- coding: utf-8 -*-

"""Source code for three rooms environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np
from omegaconf import DictConfig

from .abstract_map import AbstractMap


class ThreeRoomsMap(AbstractMap):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, size_x=25, size_y=25)
        self.locate_aisle()

    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        self.wall_matrix[np.array([1, 2, 3, 7, 8, 9, 10]), 10] = 1
        self.wall_matrix[np.array([1, 2, 3, 7, 8, 9, 10]), 14] = 1
        self.wall_matrix[
            10, np.array([1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23])
        ] = 1

        self.wall_matrix[
            14, np.array([1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23])
        ] = 1

    def locate_aisle(self):
        self.aisle_matrix[np.arange(10, 14), :] = 1
        self.aisle_matrix[:, np.arange(10, 14)] = 1
