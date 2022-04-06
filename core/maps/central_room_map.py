"""Source code for central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np

from .abstract_map import AbstractMap


class CentralRoomMap(AbstractMap):
    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        self.wall_matrix[8:17, np.asarray([8, 16])] = 1
        self.wall_matrix[np.asarray([8, 16]), 8:17] = 1

        self.wall_matrix[11:14, np.array([8, 16])] = 0
        self.wall_matrix[np.array([8, 16]), 11:14] = 0

    def set_objects_area(self):
        if self.config.type_objects == 1:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:24, 1:24] = 1
            self.objects_area_matrix[0, 8:17, 8:17] = 0

        elif self.config.type_objects == 2:
            # Set objects area for object 0
            self.objects_area_matrix[0, 1:14, 1:14] = 1
            self.objects_area_matrix[0, 11:24, 11:24] = 1
            # Set objects area for object 1
            self.objects_area_matrix[1, 11:24, 1:14] = 1
            self.objects_area_matrix[1, 1:14, 14:24] = 1

            self.objects_area_matrix[0, 8:17, 8:17] = 0
            self.objects_area_matrix[1, 8:17, 8:17] = 0
