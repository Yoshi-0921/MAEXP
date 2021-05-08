# -*- coding: utf-8 -*-

"""Source code for simple environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np
from omegaconf import DictConfig

from .abstract_map import AbstractMap


class FourRoomsMap(AbstractMap):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, size_x=24, size_y=24)
        self.locate_aisle()

    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        self.wall_matrix[
            np.array([1, 2, 3, 7, 8, 9, 10, 13, 14, 15, 16, 20, 21, 22]), 10
        ] = 1
        self.wall_matrix[
            np.array([1, 2, 3, 7, 8, 9, 10, 13, 14, 15, 16, 20, 21, 22]), 13
        ] = 1
        self.wall_matrix[
            10, np.array([1, 2, 3, 7, 8, 9, 10, 13, 14, 15, 16, 20, 21, 22])
        ] = 1
        self.wall_matrix[
            13, np.array([1, 2, 3, 7, 8, 9, 10, 13, 14, 15, 16, 20, 21, 22])
        ] = 1

    def locate_aisle(self):
        self.aisle_matrix[np.arange(10, 14), :] = 1
        self.aisle_matrix[:, np.arange(10, 14)] = 1
