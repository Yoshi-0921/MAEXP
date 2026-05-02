"""Source code for a central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import random
import torch
import numpy as np
from omegaconf import DictConfig

from .central_room_large_map import CentralRoomLargeMap


class CentralRoomLargeDestinationMap(CentralRoomLargeMap):
    def __init__(self, config: DictConfig, size_x: int, size_y: int):
        self.destination_area_vector = np.zeros(shape=(config.num_agents, 384))
        self.specified_object_types = ["0" for _ in range(config.num_agents)]
        self.type_objects = config.type_objects
        top_tensors = torch.load(config.root_dir + "top_tensor")
        bottom_tensors = torch.load(config.root_dir + "bottom_tensor")
        right_tensors = torch.load(config.root_dir + "right_tensor")
        left_tensors = torch.load(config.root_dir + "left_tensor")
        self.tensors = [left_tensors,right_tensors,top_tensors,bottom_tensors]
        super().__init__(config=config,size_x=size_x,size_y=size_y)

    def reset_destination_area(self):
        destination_area = [np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8) for _ in range(4)]
        destination_area[0][: self.SIZE_X // 2, :] = 1  # left half
        destination_area[1][self.SIZE_X // 2:, :] = 1  # right half
        destination_area[2][:, : self.SIZE_Y // 2] = 1  # top half
        destination_area[3][:, self.SIZE_Y // 2:] = 1  # bottom half

        for agent_id in range(self.num_agents):
            area_id = random.choice(range(len(destination_area)))
            object_type = random.choice(range(self.type_objects))  # Randomly assign object type
            self.destination_area_matrix[agent_id] = destination_area[area_id]
            self.destination_area_vector[agent_id] = random.choice(self.tensors[area_id])
            self.specified_object_types[agent_id] = str(object_type)
