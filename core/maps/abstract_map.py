# -*- coding: utf-8 -*-

"""Source code for environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC

import numpy as np
from omegaconf import DictConfig


class AbstractMap(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.sizeX = None
        self.sizeY = None

        self.wall_matrix = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        self.agents_matrix = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        self.objects_matrix = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        self.aisle = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)

    def reset(self):
        self.agents_matrix = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        self.objects_matrix = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)

    def reset_agents(self):
        self.agents_matrix = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)

    def reset_objects(self):
        self.objects_matrix = np.zeros((self.SIZE_X, self.SIZE_Y), dtype=np.int8)

    def coord2ind(self, position: np.array, size_x: int = None, size_y: int = None):
        pos_x, pos_y = position
        size_x = size_x or self.SIZE_X
        size_y = size_y or self.SIZE_Y

        pos_x = (size_x // 2) + pos_x
        pos_y = (size_y // 2) - pos_y
        res_pos = np.array([pos_x, pos_y], dtype=np.int8)

        return res_pos

    def ind2coord(self, position: np.array, size_x: int = None, size_y: int = None):
        pos_x, pos_y = position
        size_x = size_x or self.SIZE_X
        size_y = size_y or self.SIZE_Y

        pos_x = pos_x - (size_x // 2)
        pos_y = (size_y // 2) - pos_y
        res_pos = np.array([pos_x, pos_y], dtype=np.int8)

        return res_pos

    def locate_walls(self):
        self.wall_matrix[np.array([0, self.SIZE_X - 1]), :] = 1
        self.wall_matrix[:, np.array([0, self.SIZE_Y - 1])] = 1
