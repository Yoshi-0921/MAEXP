"""Source code for a central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np

from .abstract_map import AbstractMap


class CentralRoomComplexLargeMap(AbstractMap):
    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        # Central room walls
        self.wall_matrix[20:29, np.asarray([8, 16])] = 1
        self.wall_matrix[np.asarray([20, 28]), 8:17] = 1
        self.wall_matrix[23:26, np.array([8, 16])] = 0
        self.wall_matrix[np.array([20, 28]), 11:14] = 0

        # Additional room walls
        self.wall_matrix[20:29, np.asarray([4, 20])] = 1
        self.wall_matrix[16:21, np.asarray([8, 16])] = 1
        self.wall_matrix[28:49, np.asarray([8, 16])] = 1
        self.wall_matrix[np.asarray([16, 32]), 8:17] = 1
        self.wall_matrix[np.asarray([20, 28]), 4:9] = 1
        self.wall_matrix[np.asarray([20, 28]), 16:21] = 1
        self.wall_matrix[1:17,12] = 1
        self.wall_matrix[28:33, np.asarray([8, 16])] = 1

        # Remove walls to create passages
        self.wall_matrix[np.asarray([4,8,12]),12] = 0
        self.wall_matrix[24, np.asarray([4, 20])] = 0
        self.wall_matrix[np.asarray([20, 28]), 6] = 0
        self.wall_matrix[np.asarray([20, 28]), 18] = 0
        self.wall_matrix[np.asarray([18, 30, 36, 40, 44]), 8] = 0
        self.wall_matrix[np.asarray([18, 30, 36, 40, 44]), 16] = 0

    def set_objects_area(self):
        for object_type in range(self.type_objects):
            self.objects_area_matrix[object_type, 1:48, 1:24] = 1
            self.objects_area_matrix[object_type, 20:29, 8:17] = 0
