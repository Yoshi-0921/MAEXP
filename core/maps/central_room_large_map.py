"""Source code for a central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np

from .abstract_map import AbstractMap


class CentralRoomLargeMap(AbstractMap):
    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        self.wall_matrix[20:29, np.asarray([8, 16])] = 1
        self.wall_matrix[np.asarray([20, 28]), 8:17] = 1

        self.wall_matrix[23:26, np.array([8, 16])] = 0
        self.wall_matrix[np.array([20, 28]), 11:14] = 0

    def set_objects_area(self):
        self.objects_area_matrix[0, 1:48, 1:24] = 1
        self.objects_area_matrix[0, 20:29, 8:17] = 0
