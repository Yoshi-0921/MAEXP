# -*- coding: utf-8 -*-

"""Source code for default multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from random import random
from typing import List

import numpy as np
from core.worlds import AbstractWorld
from core.worlds.entity import Agent, Object
from omegaconf import DictConfig
from core.handlers.observations import generate_observation_handler
from .abstract_environment import AbstractEnvironment


class DefaultEnvironment(AbstractEnvironment):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        self.observation_handler = generate_observation_handler(config=config, world=world)
        self.action_space, self.observation_space = [], []
        for _ in self.agents:
            self.action_space.append(4)
            self.observation_space.append(self.observation_handler.get_observation_space)
        self.init_xys = np.asarray(config.init_xys, dtype=np.int8)

    def reset(self):
        self.objects_generated = 0
        self.objects_completed = 0
        self.agents_collided = 0
        self.walls_collided = 0
        self.world.map.reset()
        self.heatmap_agents = np.zeros(
            shape=(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int16,
        )
        self.heatmap_complete = np.zeros(
            shape=(self.num_agents, self.world.map.SIZE_X, self.world.map.SIZE_Y),
            dtype=np.int16,
        )
        self.heatmap_objects = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int16
        )
        self.heatmap_objects_left = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int16
        )
        self.heatmap_wall_collision = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int16
        )
        self.heatmap_agents_collision = np.zeros(
            shape=(self.world.map.SIZE_X, self.world.map.SIZE_Y), dtype=np.int16
        )

        for agent_id, agent in enumerate(self.agents):
            agent.collide_agents = False
            agent.collide_walls = False

            # Initialize agent position
            agent.move(self.init_xys[agent_id])

        # Initialize object position
        self.world.reset_objects()
        self.generate_objects(self.config.num_objects)

        obs_n = []
        for agent in self.agents:
            obs_n.append(self.observation_ind(agent))

        return obs_n

    def generate_objects(self, num_objects: int):
        num_generated = 0
        while num_generated < num_objects:
            x = 1 + int(random() * (self.world.map.SIZE_X - 1))
            y = 1 + int(random() * (self.world.map.SIZE_Y - 1))
            if (
                self.world.map.wall_matrix[x, y] == 0
                and self.world.map.agents_matrix[x, y] == 0
                and self.world.map.objects_matrix[x, y] == 0
                and self.world.map.aisle_matrix[x, y] == 0
            ):
                self.world.objects.append(Object())
                self.world.objects[-1].move(self.world.map.ind2coord((x, y)))
                self.world.map.objects_matrix[x, y] = 1
                self.heatmap_objects[x, y] += 1
                num_generated += 1
                self.objects_generated += 1

    def observation(self):
        raise NotImplementedError()

    def step(self, action_n: List[np.array]):
        reward_n: List[np.array] = []
        done_n: List[np.array] = []
        obs_n: List[np.array] = []

        for agent_id, agent in enumerate(self.agents):
            self.action_ind(action_n[agent_id], agent)

        # excecute action in the environment
        self.world.step()

        # obtain the outcome from the environment for each agent
        for agent_id, agent in enumerate(self.agents):
            reward_n.append(self.reward_ind(agent_id, agent))
            done_n.append(self.done_ind(agent))
            obs_n.append(self.observation_ind(agent))

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

    def reward_ind(self, agent_id: int, agent: Agent):
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0
        for obj_idx, obj in enumerate(self.world.objects):
            if all(agent.xy == obj.xy):
                reward = 1.0
                self.world.objects.pop(obj_idx)
                obj_pos_x, obj_pos_y = self.world.map.coord2ind(obj.xy)
                self.world.map.objects_matrix[obj_pos_x, obj_pos_y] = 0
                self.objects_completed += 1
                self.heatmap_complete[agent_id, obj_pos_x, obj_pos_y] += 1
                self.generate_objects(1)

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

    def done_ind(self, agent: Agent):
        for obj in self.world.objects:
            if all(agent.xy == obj.xy):
                return 1

        return 0

    def observation_ind(self, agent: Agent):

        return self.observation_handler.observation_ind(agent)
