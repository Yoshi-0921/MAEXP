import numpy as np
from core.environments.default_environment import DefaultEnvironment
from core.worlds import AbstractWorld
from omegaconf import DictConfig


class TestEnvironment(DefaultEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        # self.init_xys = np.asarray([
        #     [-24, -10], [-23, -10], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8) # default

        # self.init_xys = np.asarray([
        #     [-24, -10], [20, -8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 1-1-1 # [20, -8]
        # self.init_xys = np.asarray([
        #     [-24, -10], [20, -8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [23, -6],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 1-1-2 # [20, -8], [23, -6]
        # self.init_xys = np.asarray([
        #     [-24, -10], [20, -8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [23, -6], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 1-1-3 # [20, -8], [23, -6]

        # self.init_xys = np.asarray([
        #     [-24, -10], [-21, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 1-2-1 # [-21, 8]
        # self.init_xys = np.asarray([
        #     [-24, -10], [-21, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-18, 10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 1-2-2 # [-21, 8], [-18, 10]
        # self.init_xys = np.asarray([
        #     [-24, -10], [-21, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-18, 10], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 1-2-3 # [-21, 8], [-18, 10]

        # self.init_xys = np.asarray([
        #     [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 2-1 # [-13, 8]
        self.init_xys = np.asarray([
            [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-10, 10],
            [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        ], dtype=np.int8)  # Case 2-2 # [-13, 8], [-10, 10]

        # self.init_xys = np.asarray([
        #     [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 3-1 # [-13, 8]
        # self.init_xys = np.asarray([
        #     [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-13, 11],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 3-2 # [-13, 8] [-16: -9, 5:12]

        # self.init_xys = np.asarray([
        #     [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 4-1-1 # [-13, 8]
        # self.init_xys = np.asarray([
        #     [-9, 8], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 4-1-2 # [-13, 8], [-9, 8]
        # self.init_xys = np.asarray([
        #     [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-9, 8], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 4-1-3 # [-13, 8], [-9, 8]

        # self.init_xys = np.asarray([
        #     [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 4-2-1 # [-13, 8]
        # self.init_xys = np.asarray([
        #     [-9, 8], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-21, -11], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 4-2-2 # [-13, 8], [-13, 4]
        # self.init_xys = np.asarray([
        #     [-24, -10], [-13, 8], [-22, -10], [-21, -10], [-20, -10], [-19, -10], [-18, -10], [-17, -10], [-16, -10],
        #     [-24, -11], [-23, -11], [-22, -11], [-9, 8], [-20, -11], [-19, -11], [-18, -11], [-17, -11], [-16, -11],
        # ], dtype=np.int8)  # Case 4-2-3 # [-13, 8], [-9, 8]

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
        # object_xys = [[18, -11], [22, -8]]  # Case 1-1
        # object_xys = [[-23, 5], [-19, 8]]  # Case 1-2
        # object_xys = [[-11, 8], [-15, 8]]  # Case 2-1-1
        object_xys = [[-11, 8], [-15, 8], [-9, 5], [-9, 6], [-9, 7], [-9, 8], [-9, 9], [-9, 10], [-9, 11]]  # Case 2-1-2, Case2-2
        # object_xys = [[-11, 8], [-15, 8]]  # Case 3
        # object_xys = [[-11, 8], [-15, 5]]  # Case 4-1
        # object_xys = []  # Case 4-2
        for object_xy in object_xys:
            x, y = self.world.map.coord2ind(object_xy)
            self.world.map.objects_matrix[object_type, x, y] = 1
            self.heatmap_objects[object_type, x, y] += 1
            self.objects_generated += 1
