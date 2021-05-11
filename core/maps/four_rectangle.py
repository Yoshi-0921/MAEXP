# -*- coding: utf-8 -*-

"""Source code for rectangle environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np
from omegaconf import DictConfig

from .abstract_map import AbstractMap


class FourRectangleMap(AbstractMap):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, size_x=40, size_y=24)
        self.locate_aisle()

    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        self.wall_matrix[np.array([19, 20]), :] = 1
        self.wall_matrix[19, np.array([11, 12])] = 0
        self.wall_matrix[20, np.array([11, 12])] = 0

        self.wall_matrix[:, np.array([10, 13])] = 1
        self.wall_matrix[np.arange(8, 12), 10] = 0
        self.wall_matrix[np.arange(28, 32), 10] = 0
        self.wall_matrix[np.arange(8, 12), 13] = 0
        self.wall_matrix[np.arange(28, 32), 13] = 0

    def locate_aisle(self):
        self.aisle_matrix[:, np.arange(10, 14)] = 1
