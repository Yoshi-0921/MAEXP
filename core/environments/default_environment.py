# -*- coding: utf-8 -*-

"""Source code for default multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import random
from typing import List

import numpy as np
from core.worlds.entity import Agent, Object
from omegaconf import DictConfig

from .abstract_environment import AbstractEnvironment


class DefaultEnvironment(AbstractEnvironment):
    def __init__(self, config: DictConfig, world):
        super().__init__(config=config, world=world)
        self.visible_range = config.visible_range
        self.action_space, self.observation_space = [], []
        for agent in self.agents:
            self.action_space.append(4)
            self.observation_space.append(self.visible_range)
        self.init_xys = np.asarray(config.init_xys, dtype=np.int8)

    def reset(self):
        self.world.map.reset()
        for agent_id, agent in enumerate(self.agents):
            agent.collide_agents = False
            agent.collide_walls = False

            # Initialize agent position
            agent.move(self.init_xys[agent_id])

        # Initialize object position
        self.world.reset_objects()
        self.generate_objects()

    def generate_objects(self, num_objects: int):
        num_generated = 0
        while num_generated < num_objects:
            x = 1 + int(random() * (self.world.map.SIZE_X - 1))
            y = 1 + int(random() * (self.world.map.SIZE_Y - 1))
            if (
                self.world.map.wall_matrix[x, y] == 0
                and self.world.map.agents_matrix[x, y] == 0
                and self.world.map.objects_matrix[x, y] == 0
                and self.world.map.aisle[x, y] == 0
            ):
                self.world.objects.append(Object())
                self.world.objects[-1].move(self.world.map.ind2coord((x, y)))
                self.world.map.objects_matrix[x, y] = 1
                num_generated += 1

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
        def is_collision(agent1: Agent, agent2: Agent):
            delta_pos = agent1.xy - agent2.xy
            dist = np.sqrt(np.sum(np.square(delta_pos)))

            return True if dist == 0 else False

        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)

        for obj_idx, obj in enumerate(self.objects):
            if all(agent.xy == obj.xy):
                reward = 1.0
                self.world.objects.pop(obj_idx)
                obj_pos_x, obj_pos_y = self.world.map.coord2ind(obj.xy)
                self.world.map.objects_matrix[obj_pos_x, obj_pos_y] = 0
                self.generate_objects(1)

        # negative reward for collision with other agents
        if agent.collide_agents:
            reward = -1.0
            agent.collide_agents = False

        # negative reward for collision against walls
        if agent.collide_walls:
            reward = -1.0
            agent.collide_walls = False

        return reward

    def done_ind(self, agent: Agent):
        for obj in self.objects:
            if all(agent.xy == obj.xy):
                return 1

        return 0

    def observation_ind(self, agent: Agent):
        # 0:agents, 1:landmarks, 2:visible area
        obs = np.zeros((3, self.visible_range, self.visible_range), dtype=np.int8)
        offset = 0

        # input walls and invisible area
        obs[2, :, :] -= 1
        obs = self.fill_obs_area(obs, agent, offset, offset)

        # input objects within sight
        obs = self.fill_obs_object(obs, agent, offset, offset)

        # input agents within sight
        obs[0, self.visible_range // 2, self.visible_range // 2] = 1
        obs = self.fill_obs_agent(obs, agent, offset, offset)

        return obs

    def fill_obs_area(self, obs, agent, offset_x, offset_y):
        # 自分の場所は0
        obs[
            2,
            offset_x + self.visible_range // 2,
            offset_y + self.visible_range // 2,
        ] = 0

        for x in range(-1, 2):
            for y in [-1, 1]:
                for opr in [-1, 1]:
                    for j in range(3):
                        pos_x, pos_y = x + j * opr, y + j * y
                        local_pos_x, local_pos_y = self.world.map.coord2ind(
                            (pos_x, pos_y), self.visible_range, self.visible_range
                        )
                        pos_x, pos_y = self.world.map.coord2ind(
                            (
                                pos_x + agent.x,
                                pos_y + agent.y,
                            )
                        )
                        # 場外なら-1
                        if (
                            pos_x < 0
                            or self.world.map.SIZE_X <= pos_x
                            or pos_y < 0
                            or self.world.map.SIZE_Y <= pos_y
                        ):
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            break
                        # セルが真ん中で壁のない角の方向なら続ける
                        if (
                            j == 0
                            and x == 0
                            and self.world.map.matrix[pos_x + opr, pos_y, 0] == 0
                        ):
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0
                            continue
                        # 壁なら-1
                        if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            break
                        # セルが角で真ん中に壁があるならbreak
                        if (
                            j == 0
                            and x != 0
                            and self.world.map.matrix[pos_x - x, pos_y, 0] == 1
                            and opr != x
                        ):
                            break
                        # 何もないなら0
                        else:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0

        for y in range(-1, 2):
            for x in [-1, 1]:
                for opr in range(-1, 2):
                    for j in range(3):
                        pos_x, pos_y = x + j * x, y + j * opr
                        local_pos_x, local_pos_y = self.world.map.coord2ind(
                            (pos_x, pos_y), self.visible_range, self.visible_range
                        )
                        pos_x, pos_y = self.world.map.coord2ind(
                            (
                                pos_x + agent.x,
                                pos_y + agent.y,
                            )
                        )
                        # 場外なら-1
                        if (
                            pos_x < 0
                            or self.world.map.SIZE_X <= pos_x
                            or pos_y < 0
                            or self.world.map.SIZE_Y <= pos_y
                        ):
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            break
                        # セルが真ん中で壁のない角の方向なら続ける
                        if (
                            j == 0
                            and y == 0
                            and self.world.map.matrix[pos_x, pos_y - opr, 0] == 0
                        ):
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0
                            continue
                        # 壁なら-1
                        if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            break
                        # セルが角で真ん中に壁があるならbreak
                        if (
                            j == 0
                            and y != 0
                            and self.world.map.matrix[pos_x, pos_y + y, 0] == 1
                            and opr != y
                        ):
                            break
                        # 何もないなら0
                        else:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0

        for opr_x in [-1, 1]:
            for x in range(self.visible_range // 2 + 1):
                # 壁ならbreak
                pos_x, pos_y = (
                    agent.x + (x * opr_x),
                    agent.y,
                )
                pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                local_pos_x, local_pos_y = self.world.map.coord2ind(
                    ((x * opr_x), 0), self.visible_range, self.visible_range
                )
                if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                    obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                    break
                for opr_y in [-1, 1]:
                    for y in range(2):
                        pos_x, pos_y = (
                            agent.x + (x * opr_x),
                            agent.y + (y * opr_y),
                        )
                        pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                        local_pos_x, local_pos_y = self.world.map.coord2ind(
                            ((x * opr_x), (y * opr_y)),
                            self.visible_range,
                            self.visible_range,
                        )
                        # 場外なら-1
                        if (
                            pos_x < 0
                            or self.world.map.SIZE_X <= pos_x
                            or pos_y < 0
                            or self.world.map.SIZE_Y <= pos_y
                        ):
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            continue
                        # 壁なら-1
                        if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            break
                        # 何もないなら0
                        else:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0

        for opr_y in [-1, 1]:
            for y in range(self.visible_range // 2 + 1):
                # 壁ならbreak
                pos_x, pos_y = agent.x, agent.y + (y * opr_y)
                pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                local_pos_x, local_pos_y = self.world.map.coord2ind(
                    (0, (y * opr_y)), self.visible_range, self.visible_range
                )
                if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                    obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                    break
                for opr_x in [-1, 1]:
                    for x in range(2):
                        pos_x, pos_y = (
                            agent.x + (x * opr_x),
                            agent.y + (y * opr_y),
                        )
                        pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                        local_pos_x, local_pos_y = self.world.map.coord2ind(
                            ((x * opr_x), (y * opr_y)),
                            self.visible_range,
                            self.visible_range,
                        )
                        # 場外なら-1
                        if (
                            pos_x < 0
                            or self.world.map.SIZE_X <= pos_x
                            or pos_y < 0
                            or self.world.map.SIZE_Y <= pos_y
                        ):
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            continue
                        # 壁なら-1
                        if self.world.map.matrix[pos_x, pos_y, 0] == 1:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = -1
                            break
                        # 何もないなら0
                        else:
                            obs[2, offset_x + local_pos_x, offset_y + local_pos_y] = 0

        return obs

    def fill_obs_agent(self, obs, agent, offset_x, offset_y):
        for a in self.agents:
            diff_x, diff_y = a.xy - agent.xy
            if abs(diff_x) > 3 or abs(diff_y) > 3 or (diff_x == 0 and diff_y == 0):
                continue

            pos_x, pos_y = self.world.map.coord2ind(
                position=(a.x - agent.x, a.y - agent.y),
                size_x=self.visible_range,
                size_y=self.visible_range,
            )
            # 見える範囲なら追加
            if obs[2, offset_x + pos_x, offset_y + pos_y] != -1:
                obs[0, offset_x + pos_x, offset_y + pos_y] = 1

        return obs

    def fill_obs_object(self, obs, agent, offset_x, offset_y):
        for obj in self.objects:
            if abs(obj.x - agent.x) > 3 or abs(obj.y - agent.y) > 3:
                continue

            pos_x, pos_y = self.world.map.coord2ind(
                position=(obj.x - agent.x, obj.y - agent.y),
                size_x=self.visible_range,
                size_y=self.visible_range,
            )
            # 見える範囲なら追加
            if obs[2, offset_x + pos_x, offset_y + pos_y] != -1:
                obs[1, offset_x + pos_x, offset_y + pos_y] = 1

        return obs
