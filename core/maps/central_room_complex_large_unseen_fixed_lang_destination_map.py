"""Source code for a central room environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
import random
import torch
import numpy as np
from omegaconf import DictConfig

from .central_room_complex_large_map import CentralRoomComplexLargeMap


class CentralRoomComplexLargeUnseenFixedLangDestinationMap(CentralRoomComplexLargeMap):
    def __init__(self, config: DictConfig, size_x: int, size_y: int):
        self.destination_area_vector = np.zeros(shape=(config.num_agents, 384))
        self.specified_object_types = ["0" for _ in range(config.num_agents)]
        self.type_objects = config.type_objects
        unseen_type0_left_tensors = torch.load(config.map.root_dir + "unseen_type0_left_tensor")
        unseen_type0_right_tensors = torch.load(config.map.root_dir + "unseen_type0_right_tensor")
        unseen_type0_top_tensors = torch.load(config.map.root_dir + "unseen_type0_top_tensor")
        unseen_type0_bottom_tensors = torch.load(config.map.root_dir + "unseen_type0_bottom_tensor")
        unseen_type1_top_tensors = torch.load(config.map.root_dir + "unseen_type1_top_tensor")
        unseen_type1_bottom_tensors = torch.load(config.map.root_dir + "unseen_type1_bottom_tensor")
        unseen_type1_right_tensors = torch.load(config.map.root_dir + "unseen_type1_right_tensor")
        unseen_type1_left_tensors = torch.load(config.map.root_dir + "unseen_type1_left_tensor")
        self.tensors = [[unseen_type0_left_tensors,unseen_type0_right_tensors,unseen_type0_top_tensors,unseen_type0_bottom_tensors],
                        [unseen_type1_left_tensors,unseen_type1_right_tensors,unseen_type1_top_tensors,unseen_type1_bottom_tensors]]
        super().__init__(config=config,size_x=size_x,size_y=size_y)

    def reset_destination_area(self):
        destination_area = [np.zeros(shape=(self.SIZE_X, self.SIZE_Y), dtype=np.int8) for _ in range(4)]
        destination_area[0][: self.SIZE_X // 2, :] = 1  # left half
        destination_area[1][self.SIZE_X // 2:, :] = 1  # right half
        destination_area[2][:, : self.SIZE_Y // 2] = 1  # top half
        destination_area[3][:, self.SIZE_Y // 2:] = 1  # bottom half

        for agent_id in range(self.num_agents):
            area_id = 0
            object_type = 0
            if agent_id >= 4:
                area_id = 1
                object_type = 1
            self.destination_area_matrix[agent_id] = destination_area[area_id]
            self.destination_area_vector[agent_id] = random.choice(self.tensors[object_type][area_id])
            self.specified_object_types[agent_id] = str(object_type)
