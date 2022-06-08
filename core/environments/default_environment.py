"""Source code for default multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import random
from typing import List

import numpy as np
from core.handlers.observations import generate_observation_handler
from core.worlds import AbstractWorld
from core.worlds.entity import Agent
from omegaconf import DictConfig

from .abstract_environment import AbstractEnvironment


class DefaultEnvironment(AbstractEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        self.observation_handler = generate_observation_handler(
            config=config, world=self.world
        )
        self.action_space, self.observation_space = [], []
        for _ in self.agents:
            self.action_space.append(4)
            self.observation_space.append(self.observation_handler.observation_space)
        self.init_xys = np.asarray(config.init_xys, dtype=np.int8)
        self.init_xys_order = [i for i in range(len(self.init_xys))]
        self.type_objects = config.type_objects
        self.agent_tasks = config.agent_tasks

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
        self.current_step = 0

    def reset(self):
        self.objects_generated = 0
        self.objects_completed = 0
        self.agents_collided = 0
        self.walls_collided = 0
        self.current_step = 0
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

        if self.config.shuffle_init_xys:
            random.shuffle(self.init_xys_order)

        for agent_id, (agent, order) in enumerate(
            zip(self.agents, self.init_xys_order)
        ):
            agent.collide_agents = False
            agent.collide_walls = False

            # Initialize agent position
            agent.move(self.init_xys[order].copy())
            pos_x, pos_y = self.world.map.coord2ind(self.init_xys[order])
            self.world.map.agents_matrix[agent_id, pos_x, pos_y] = 1

        # Initialize object position
        self.generate_objects()

        obs_n = self.observation_handler.reset(self.agents)

        return obs_n

    def generate_objects(self, num_objects: int = None, object_type: int = None):
        num_objects = num_objects or self.config.num_objects
        if object_type is None:
            for object_type in range(self.type_objects):
                self._generate_objects(
                    num_objects,
                    object_type=object_type,
                )
        else:
            self._generate_objects(num_objects, object_type)

    def _generate_objects(self, num_objects: int, object_type: int = 0):
        num_generated = 0
        while num_generated < num_objects:
            x = 1 + int(random.random() * (self.world.map.SIZE_X - 1))
            y = 1 + int(random.random() * (self.world.map.SIZE_Y - 1))
            if (
                self.world.map.wall_matrix[x, y] == 0
                and self.world.map.agents_matrix[:, x, y].sum() == 0
                and self.world.map.objects_matrix[:, x, y].sum() == 0
                and self.world.map.objects_area_matrix[object_type, x, y] == 1
            ):
                self.world.map.objects_matrix[object_type, x, y] = 1
                self.heatmap_objects[object_type, x, y] += 1
                num_generated += 1
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

        self.current_step += 1
        self.heatmap_objects_left += self.world.map.objects_matrix

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
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0
        if self.agent_tasks[agent_id] == "-1":
            return reward

        for object_type in self.agent_tasks[agent_id]:
            if (
                self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y]
                == self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y]
                == 1
            ):
                reward = 1.0
                self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y] = 0
                self.objects_completed += 1
                self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1
                if self.config.keep_objects_num:
                    self.generate_objects(1, int(object_type))

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
        # for obj in self.world.objects:
        #     if all(agent.xy == obj.xy):
        #         return 1
        if self.world.map.objects_matrix.sum() or self.current_step == (self.config.max_episode_length - 1):
            return True

        return False

    def observation_ind(self, agents: List[Agent], agent: Agent, agent_id: int):

        return self.observation_handler.observation_ind(agents, agent, agent_id)

    def accumulate_heatmap(self):
        self.heatmap_accumulated_agents += self.heatmap_agents
        self.heatmap_accumulated_complete += self.heatmap_complete
        self.heatmap_accumulated_objects += self.heatmap_objects
        self.heatmap_accumulated_objects_left += self.heatmap_objects_left
        self.heatmap_accumulated_wall_collision += self.heatmap_wall_collision
        self.heatmap_accumulated_agents_collision += self.heatmap_agents_collision
