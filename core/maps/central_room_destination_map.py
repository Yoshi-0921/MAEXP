"""Source code for central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import numpy as np
import random
from .central_room_map import CentralRoomMap


class CentralRoomDestinationMap(CentralRoomMap):
    def reset_destination_area(self):
        destination_area = [np.ones(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8) for _ in range(5)]
        destination_area[0][: self.SIZE_X // 2, :] = 0
        destination_area[1][self.SIZE_X // 2:, :] = 0
        destination_area[2][:, : self.SIZE_Y // 2] = 0
        destination_area[3][:, self.SIZE_Y // 2:] = 0
        random.shuffle(destination_area)

        for agent_id in range(self.num_agents):
            self.destination_area_matrix[agent_id] = random.choice(destination_area)
