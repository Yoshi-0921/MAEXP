import random
from typing import List

import numpy as np
from omegaconf import DictConfig

from core.handlers.observations import generate_observation_handler
from core.worlds import AbstractWorld
from core.worlds.entity import Agent

from .abstract_environment import AbstractEnvironment
from .default_environment import DefaultEnvironment

from .sequential_environment import SequentialEnvironment


class TestSequentialEnvironment(SequentialEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        agent_xys = [
            [4,9],[5,9],[7,7],[6,9],[7,9],[8,9],[9,9],[10,9]
            # [-3,2],[-2,2],[0,0],[-1,2],[0,2],[1,2],[2,2],[3,2]
            # [-10,-9],[-9,-9],[-7,-7],[-8,-9],[-7,-9],[-6,-9],[-5,-9],[-4,-9]
            ]
        for agent_id, agent_xy in enumerate(agent_xys):
            self.init_xys[agent_id] = np.asarray(agent_xy, dtype=np.int8)

    def generate_objects(self, num_objects: int = None, object_type: int = None):
        object_types = [0,1,2]
        object_xys = [
            [6, 4], [7, 4], [8,4]
            # [-1, -3], [0, -3], [1,-3]
            # [-8, -4], [-7, -4], [-6,-4]
            ]
        for object_type, object_xy in zip(object_types, object_xys):
            x, y = self.world.map.coord2ind(object_xy)
            self.world.map.objects_matrix[object_type, x, y] = 1
            self.heatmap_objects[object_type, x, y] += 1
            self.objects_generated += 1
