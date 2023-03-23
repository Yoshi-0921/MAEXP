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
