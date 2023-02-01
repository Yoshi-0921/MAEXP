import numpy as np
from core.environments.default_environment import DefaultEnvironment
from core.worlds import AbstractWorld
from omegaconf import DictConfig


class TestEnvironment(DefaultEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        # self.init_xys = np.asarray([
        #     [8, -10], [9, -10], [10, -10], [11, -10], [8, -11], [9, -11], [10, -11], [11, -11],
        # ], dtype=np.int8) # default right bottom
        self.init_xys = np.asarray([
            [-8, 10], [-9, 10], [-10, 10], [-11, 10], [-8, 11], [-9, 11], [-10, 11], [-11, 11],
        ], dtype=np.int8) # default left up
        # self.init_xys = np.asarray([
        #     [8, -10], [9, -10], [10, -10], [11, -10], [8, 8], [9, -11], [10, -11], [11, -11],
        # ], dtype=np.int8) # right up
        # self.init_xys = np.asarray([
        #     [8, -10], [9, -10], [11, 10], [11, -10], [8, 8], [9, -11], [10, -11], [11, -11],
        # ], dtype=np.int8) # right up with

        # right up
        # self.init_xys[3] = np.asarray([8, 8], dtype=np.int8)
        # right up with
        # self.init_xys[4] = np.asarray([11, 10], dtype=np.int8)

         # left up
        # self.init_xys[1] = np.asarray([-2, 8], dtype=np.int8)
        # left up with
        # self.init_xys[4] = np.asarray([1, 10], dtype=np.int8)

        # right bottom
        # self.init_xys[1] = np.asarray([2, -8], dtype=np.int8)
        # right bottom with
        # self.init_xys[4] = np.asarray([5, -6], dtype=np.int8)

        # left bottom
        # self.init_xys[3] = np.asarray([-8, -8], dtype=np.int8)
        # right bottom with
        # self.init_xys[4] = np.asarray([-5, -6], dtype=np.int8)

        # right up objects
        self.init_xys[3] = np.asarray([5, 8], dtype=np.int8)
        self.init_xys[4] = np.asarray([8, 10], dtype=np.int8)


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
        object_type = 1
        # object_xys = [[10, 8], [6, 5]]  # right up
        # object_xys = [[-10, 5], [-6, 8]]  # left up
        # object_xys = [[-10, -11], [-6, -8]] # left bottom
        # object_xys = [[10, -8], [6, -11]]  # right bottom

        # object_xys = [[3, 8], [7, 8]] # original objects
        object_xys = [[3, 8], [7, 8], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11]] # more objects

        for object_xy in object_xys:
            x, y = self.world.map.coord2ind(object_xy)
            self.world.map.objects_matrix[object_type, x, y] = 1
            self.heatmap_objects[object_type, x, y] += 1
            self.objects_generated += 1
