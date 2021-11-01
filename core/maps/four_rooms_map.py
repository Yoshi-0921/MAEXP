# -*- coding: utf-8 -*-

"""Source code for simple environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np

from .abstract_map import AbstractMap


class FourRoomsMap(AbstractMap):
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
