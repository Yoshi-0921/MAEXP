# -*- coding: utf-8 -*-

"""Source code for rectangle environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np

from .abstract_map import AbstractMap


class FourRectangleMap(AbstractMap):
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
