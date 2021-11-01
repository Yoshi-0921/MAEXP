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

    def set_objects_area(self):
        if self.config.type_objects == 1:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1: 19, 1: 10] = 1
            self.objects_area_matrix[0, 1: 19, 14: self.SIZE_Y - 1] = 1
            self.objects_area_matrix[0, 21: self.SIZE_X - 1, 1: 10] = 1
            self.objects_area_matrix[0, 21: self.SIZE_X - 1, 14: self.SIZE_Y - 1] = 1

        elif self.config.type_objects == 2:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1: 19, 1: 10] = 1
            self.objects_area_matrix[0, 1: 19, 14: self.SIZE_Y - 1] = 1
            # Set objects area for object 1
            self.objects_area_matrix[1, 21: self.SIZE_X - 1, 1: 10] = 1
            self.objects_area_matrix[1, 21: self.SIZE_X - 1, 14: self.SIZE_Y - 1] = 1
