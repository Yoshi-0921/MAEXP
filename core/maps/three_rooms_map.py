"""Source code for three rooms environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import random

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

    def reset_destination_area(self):
        destination_area = [np.ones(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8) for _ in range(9)]
        destination_area[0][: self.SIZE_X // 2, :] = 0
        destination_area[1][: self.SIZE_X // 2, :] = 0
        destination_area[2][self.SIZE_X // 2:, :] = 0
        destination_area[3][self.SIZE_X // 2:, :] = 0
        destination_area[4][:, : self.SIZE_X // 2] = 0
        destination_area[5][:, : self.SIZE_X // 2] = 0
        destination_area[6][:, self.SIZE_X // 2:] = 0
        destination_area[7][:, self.SIZE_X // 2:] = 0
        random.shuffle(destination_area)

        for agent_id in range(self.num_agents):
            self.destination_area_matrix[agent_id] = random.choice(destination_area)
