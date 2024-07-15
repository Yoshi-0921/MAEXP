import numpy as np
from omegaconf import DictConfig

from core.environments.default_environment import DefaultEnvironment
from core.worlds import AbstractWorld


class TestEnvironment(DefaultEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        # right up objects
        self.init_xys[1] = np.asarray([-8, 8], dtype=np.int8)
        self.init_xys[4] = np.asarray([-11, 9], dtype=np.int8)

    def reset(self):
        self.objects_generated = 0
        self.objects_completed = 0
        self.agents_collided = 0
        self.walls_collided = 0
        self.world.map.reset()
        self.heatmap_agents = np.zeros(
            shape=(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_complete = np.zeros(
            shape=(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_objects = np.zeros(
            shape=(self.type_objects, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_objects_left = np.zeros(
            shape=(self.type_objects, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_wall_collision = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int32
        )
        self.heatmap_agents_collision = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int32
        )

        for agent_id, agent in enumerate(self.agents):
            agent.collide_agents = False
            agent.collide_walls = False

            # Initialize agent position
            agent.move(self.init_xys[agent_id])
            pos_x, pos_y = self.world.map.coord2ind(self.init_xys[agent_id])
            self.world.map.agents_matrix[agent_id, pos_x, pos_y] = 1

        # Initialize object position
        self.generate_test_objects()

        obs_n = self.observation_handler.reset(self.agents)

        return obs_n

    def generate_objects(self, num_objects: int = None, object_type: int = None):
        pass

    def generate_test_objects(self):
        object_type = 0
        object_xys = [[-10, 8], [-6, 6]]
        # object_xys = [[3, 8], [7, 8], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11]]  # more objects

        for object_xy in object_xys:
            x, y = self.world.map.coord2ind(object_xy)
            self.world.map.objects_matrix[object_type, x, y] = 1
            self.heatmap_objects[object_type, x, y] += 1
            self.objects_generated += 1
