"""Source code for two central rooms environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np

from .abstract_map import AbstractMap


class TwoCentralRoomsMap(AbstractMap):
    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        self.wall_matrix[8:17, np.asarray([8, 16])] = 1
        self.wall_matrix[np.asarray([8, 16]), 8:17] = 1

        self.wall_matrix[11:14, np.array([8, 16])] = 0
        self.wall_matrix[np.array([8, 16]), 11:14] = 0

        self.wall_matrix[33:42, np.asarray([8, 16])] = 1
        self.wall_matrix[np.asarray([33, 41]), 8:17] = 1

        self.wall_matrix[36:39, np.array([8, 16])] = 0
        self.wall_matrix[np.array([33, 41]), 11:14] = 0

    def set_objects_area(self):
        if self.config.type_objects == 1:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:24, 1:24] = 1
            self.objects_area_matrix[0, 8:17, 8:17] = 0

        elif self.config.type_objects == 2:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:26, 1:13] = 1
            self.objects_area_matrix[0, 8:17, 8:13] = 0

            self.objects_area_matrix[0, 24:49, 12:24] = 1
            self.objects_area_matrix[0, 33:42, 12:17] = 0

            # Set objects area for object 1
            self.objects_area_matrix[1, 1:26, 12:24] = 1
            self.objects_area_matrix[1, 8:17, 12:17] = 0

            self.objects_area_matrix[1, 24:49, 1:13] = 1
            self.objects_area_matrix[1, 33:42, 8:13] = 0
