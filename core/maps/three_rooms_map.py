# -*- coding: utf-8 -*-

"""Source code for three rooms environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np

from .abstract_map import AbstractMap


class ThreeRoomsMap(AbstractMap):
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

    def set_objects_area(self):
        if self.config.type_objects == 1:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:10, 1:10] = 1
            self.objects_area_matrix[0, 1:10, 15:24] = 1
            self.objects_area_matrix[0, 15:24, 1:24] = 1

        elif self.config.type_objects == 2:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:10, 1:10] = 1
            self.objects_area_matrix[0, 15:24, 1:24] = 1
            # Set objects area for object 1
            self.objects_area_matrix[1, 1:10, 15:24] = 1
            self.objects_area_matrix[1, 15:24, 1:24] = 1
