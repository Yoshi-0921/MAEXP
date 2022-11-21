"""Source code for dynamic simple environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import numpy as np
from omegaconf import DictConfig

from .simple_map import SimpleMap


class DynamicSimpleMap(SimpleMap):
    def __init__(self, config: DictConfig, size_x: int, size_y: int):
        self.obstacles_pos = [
            [size_x // 2, 1],
            [size_x // 2, size_y // 2],
        ]
        self.obstacles_homing_flag = [False, False]
        super().__init__(config, size_x, size_y)

    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1

        for (obstacle_pos_x, obstacle_pos_y) in self.obstacles_pos:
            self.wall_matrix[obstacle_pos_x, obstacle_pos_y] = 1

    def step(self):
        for (obstacle_pos_x, obstacle_pos_y) in self.obstacles_pos:
            self.wall_matrix[obstacle_pos_x, obstacle_pos_y] = 0

        for obstacle_id, (obstacle_pos_x, obstacle_pos_y) in enumerate(
            self.obstacles_pos
        ):
            if (
                obstacle_pos_x == self.SIZE_X // 2 and obstacle_pos_y == self.SIZE_Y - 2 and obstacle_id == 0
            ) or (
                obstacle_pos_x == self.SIZE_X - 2 and obstacle_pos_y == self.SIZE_Y // 2 and obstacle_id == 1
            ):
                self.obstacles_homing_flag[obstacle_id] = True

            elif (
                obstacle_pos_x == self.SIZE_X // 2 and obstacle_pos_y == 1 and obstacle_id == 0
            ) or (
                obstacle_pos_x == 1 and obstacle_pos_y == self.SIZE_Y // 2 and obstacle_id == 1
            ):
                self.obstacles_homing_flag[obstacle_id] = False

        if self.obstacles_homing_flag[0]:
            obstacle_pos_x, obstacle_pos_y = self.obstacles_pos[0]
            self.obstacles_pos[0] = [obstacle_pos_x, obstacle_pos_y - 1]
        else:
            obstacle_pos_x, obstacle_pos_y = self.obstacles_pos[0]
            self.obstacles_pos[0] = [obstacle_pos_x, obstacle_pos_y + 1]

        if self.obstacles_homing_flag[1]:
            obstacle_pos_x, obstacle_pos_y = self.obstacles_pos[1]
            self.obstacles_pos[1] = [obstacle_pos_x - 1, obstacle_pos_y]
        else:
            obstacle_pos_x, obstacle_pos_y = self.obstacles_pos[1]
            self.obstacles_pos[1] = [obstacle_pos_x + 1, obstacle_pos_y]

        for (obstacle_pos_x, obstacle_pos_y) in self.obstacles_pos:
            self.wall_matrix[obstacle_pos_x, obstacle_pos_y] = 1
