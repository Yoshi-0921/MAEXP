"""Source code for a central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np
import random
from .central_room_large_map import CentralRoomLargeMap


class CentralRoomLargeDestinationMap(CentralRoomLargeMap):
    def reset_destination_area(self):
        destination_area = [np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8) for _ in range(4)]
        destination_area[0][: self.SIZE_X // 2, :] = 1
        destination_area[1][self.SIZE_X // 2:, :] = 1
        destination_area[2][:, : self.SIZE_Y // 2] = 1
        destination_area[3][:, self.SIZE_Y // 2:] = 1
        # destination_area[0][: self.SIZE_X // 2, : self.SIZE_Y // 2] = 1
        # destination_area[1][self.SIZE_X // 2:, : self.SIZE_Y // 2] = 1
        # destination_area[2][: self.SIZE_X // 2, self.SIZE_Y // 2:] = 1
        # destination_area[3][self.SIZE_X // 2:, self.SIZE_Y // 2:] = 1
        random.shuffle(destination_area)

        for agent_id in range(self.num_agents):
            self.destination_area_matrix[agent_id] = random.choice(destination_area)

        # half 1
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[: self.SIZE_X // 2, :] = 1
        # self.destination_area_matrix[0] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :] = 1
        # self.destination_area_matrix[1] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, : self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[2] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[3] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[: self.SIZE_X // 2, :] = 1
        # self.destination_area_matrix[4] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :] = 1
        # self.destination_area_matrix[5] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, : self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[6] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[7] = destination_area

        # half 2
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :] = 1
        # self.destination_area_matrix[0] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[: self.SIZE_X // 2, :] = 1
        # self.destination_area_matrix[1] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[2] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, : self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[3] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :] = 1
        # self.destination_area_matrix[4] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[: self.SIZE_X // 2, :] = 1
        # self.destination_area_matrix[5] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[6] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:, : self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[7] = destination_area

        # quarter 1
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[0] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[1] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[2] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[3] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[4] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[5] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[6] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[7] = destination_area

        # quarter 2
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[0] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[1] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[2] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[3] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, :self.SIZE_Y // 2] = 1
        # self.destination_area_matrix[4] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[5] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[:self.SIZE_X // 2, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[6] = destination_area
        # destination_area = np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)
        # destination_area[self.SIZE_X // 2:, self.SIZE_Y // 2:] = 1
        # self.destination_area_matrix[7] = destination_area
