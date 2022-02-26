from typing import List

import numpy as np
from core.worlds import AbstractWorld
from core.worlds.entity import Agent, Object
from omegaconf import DictConfig
from core.handlers.observations import generate_observation_handler
from .abstract_environment import AbstractEnvironment


class TypesTestEnvironment(AbstractEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        self.observation_handler = generate_observation_handler(
            config=config, world=self.world
        )
        self.action_space, self.observation_space = [], []
        for _ in self.agents:
            self.action_space.append(4)
            self.observation_space.append(self.observation_handler.observation_space)
        # self.init_xys = np.asarray([[-8, -8], [3, 3], [-8, -7], [-7, -8], [-6, -4], [-7, -7]], dtype=np.int8) # Situation 1
        # self.init_xys = np.asarray([[6, 4], [3, 3], [-8, -7], [-7, -8], [-6, -4], [-7, -7]], dtype=np.int8) # Situation 2
        # self.init_xys = np.asarray([[-8, -8], [3, 3], [-8, -7], [-7, -8], [-6, -4], [6, 4]], dtype=np.int8) # Situation 3
        # self.init_xys = np.asarray([[0, 2], [3, 3], [-8, -7], [-7, -8], [-6, -4], [6, 4]], dtype=np.int8) # Situation 4
        self.init_xys = np.asarray([[-8, -8], [3, 3], [-8, -7], [-7, -8], [-6, -4], [-7, -7]], dtype=np.int8)  # おまけ
        self.type_objects = config.type_objects

        self.heatmap_accumulated_agents = np.zeros(
            shape=(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_accumulated_complete = np.zeros(
            shape=(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_accumulated_objects = np.zeros(
            shape=(self.type_objects, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_accumulated_objects_left = np.zeros(
            shape=(self.type_objects, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int32,
        )
        self.heatmap_accumulated_wall_collision = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int32
        )
        self.heatmap_accumulated_agents_collision = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int32
        )

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
        self.world.reset_objects()
        self.generate_test_objects()

        obs_n = self.observation_handler.reset(self.agents)

        return obs_n

    def generate_test_objects(self):
        x = 11
        y = 9
        if (
            self.world.map.wall_matrix[x, y] == 0
            and self.world.map.agents_matrix[:, x, y].sum() == 0
            and self.world.map.objects_matrix[0, x, y] == 0
            and self.world.map.objects_area_matrix[0, x, y] == 1
        ):
            self.world.objects.append(Object())
            self.world.objects[-1].move(self.world.map.ind2coord((x, y)))
            self.world.map.objects_matrix[0, x, y] = 1
            self.heatmap_objects[0, x, y] += 1
            self.objects_generated += 1

        x = 15
        y = 5
        if (
            self.world.map.wall_matrix[x, y] == 0
            and self.world.map.agents_matrix[:, x, y].sum() == 0
            and self.world.map.objects_matrix[0, x, y] == 0
            and self.world.map.objects_area_matrix[0, x, y] == 1
        ):
            self.world.objects.append(Object())
            self.world.objects[-1].move(self.world.map.ind2coord((x, y)))
            self.world.map.objects_matrix[0, x, y] = 1
            self.heatmap_objects[0, x, y] += 1
            self.objects_generated += 1

    def observation(self):
        raise NotImplementedError()

    def step(self, action_n: List[np.array], order: List[int]):
        reward_n: List[np.array] = []
        done_n: List[np.array] = []
        obs_n: List[np.array] = []

        for agent_id, agent in enumerate(self.agents):
            self.action_ind(action_n[agent_id], agent)

        # excecute action in the environment
        self.world.step(order)

        # obtain the outcome from the environment for each agent
        for agent_id, agent in enumerate(self.agents):
            reward_n.append(self.reward_ind(self.agents, agent, agent_id))
            done_n.append(self.done_ind(self.agents, agent, agent_id))
            obs_n.append(self.observation_ind(self.agents, agent, agent_id))

        self.observation_handler.step(self.agents)

        self.heatmap_objects_left += self.world.map.objects_matrix[0]

        return reward_n, done_n, obs_n

    def reward(self):
        raise NotImplementedError()

    def action_ind(self, action: int, agent: Agent):
        if action == 0:
            agent.action = np.array([1, 0], dtype=np.int8)

        elif action == 1:
            agent.action = np.array([0, 1], dtype=np.int8)

        elif action == 2:
            agent.action = np.array([-1, 0], dtype=np.int8)

        elif action == 3:
            agent.action = np.array([0, -1], dtype=np.int8)

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        if agent_id in [4, 5]:
            return 0.0
        else:
            a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
            self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

            reward = 0.0
            for obj_idx, obj in enumerate(self.world.objects):
                if all(agent.xy == obj.xy):
                    reward = 1.0
                    self.world.objects.pop(obj_idx)
                    obj_pos_x, obj_pos_y = self.world.map.coord2ind(obj.xy)
                    self.world.map.objects_matrix[0, obj_pos_x, obj_pos_y] = 0
                    self.objects_completed += 1
                    self.heatmap_complete[agent_id, obj_pos_x, obj_pos_y] += 1
                    # self.generate_objects(1)

            # negative reward for collision with other agents
            if agent.collide_agents:
                reward = -1.0
                agent.collide_agents = False
                self.heatmap_agents_collision[a_pos_x, a_pos_y] += 1
                self.agents_collided += 1

            # negative reward for collision against walls
            if agent.collide_walls:
                reward = -1.0
                agent.collide_walls = False
                self.heatmap_wall_collision[a_pos_x, a_pos_y] += 1
                self.walls_collided += 1

            return reward

    def done_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        for obj in self.world.objects:
            if all(agent.xy == obj.xy):
                return 1

        return 0

    def observation_ind(self, agents: List[Agent], agent: Agent, agent_id: int):

        return self.observation_handler.observation_ind(agents, agent, agent_id)

    def accumulate_heatmap(self):
        self.heatmap_accumulated_agents += self.heatmap_agents
        self.heatmap_accumulated_complete += self.heatmap_complete
        self.heatmap_accumulated_objects += self.heatmap_objects
        self.heatmap_accumulated_objects_left += self.heatmap_objects_left
        self.heatmap_accumulated_wall_collision += self.heatmap_wall_collision
        self.heatmap_accumulated_agents_collision += self.heatmap_agents_collision
