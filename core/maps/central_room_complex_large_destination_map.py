"""Source code for a central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import random
import numpy as np

from .central_room_complex_large_map import CentralRoomComplexLargeMap


class CentralRoomComplexLargeDestinationMap(CentralRoomComplexLargeMap):
    def reset_destination_area(self):
        destination_area = [np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8) for _ in range(5)]
        destination_area[0][: self.SIZE_X // 2, :] = 1  # left half
        destination_area[1][self.SIZE_X // 2:, :] = 1  # right half
        destination_area[2][:, : self.SIZE_Y // 2] = 1  # top half
        destination_area[3][:, self.SIZE_Y // 2:] = 1  # bottom half
        destination_area[4] = np.ones(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8)

        for agent_id in range(self.num_agents):
            area_id = random.choice(range(len(destination_area)))
            self.destination_area_matrix[agent_id] = destination_area[area_id]
