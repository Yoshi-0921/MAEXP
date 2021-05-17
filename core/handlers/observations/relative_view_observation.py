# -*- coding: utf-8 -*-

"""Source code for observation handler using the local view method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from .abstract_observation import AbstractObservation
from core.worlds.entity import Agent
import numpy as np


class RelativeViewObservaton(AbstractObservation):
    @property
    def get_observation_space(self):
        return [4, self.world.map.SIZE_X, self.world.map.SIZE_Y]

    def observation_ind(self, agent: Agent):
        # 0: me, 1:agents, 2:landmarks, 3:visible area
        obs = np.zeros(self.get_observation_space, dtype=np.int8)
        offset = 0

        # input walls and invisible area
        obs = self.fill_obs_area(obs, agent, offset, offset)

        # input objects within sight
        obs = self.fill_obs_object(obs, agent, offset, offset)

        # input agents within sight
        obs = self.fill_obs_agent(obs, agent, offset, offset)

        return obs

    def fill_obs_area(self, obs, agent, offset_x, offset_y):
        obs[3, :, :] -= 1
        # 自分の場所は0
        obs[3, agent.x, agent.y] = 0

        for x in range(-1, 2):
            for y in [-1, 1]:
                for opr in [-1, 1]:
                    for j in range(3):
                        pos_x, pos_y = x + j * opr, y + j * y
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
                            obs[3, pos_x, pos_y] = -1
                            break
                        # セルが真ん中で壁のない角の方向なら続ける
                        if (
                            j == 0
                            and x == 0
                            and self.world.map.wall_matrix[pos_x + opr, pos_y] == 0
                        ):
                            obs[3, pos_x, pos_y] = 0
                            continue
                        # 壁なら-1
                        if self.world.map.wall_matrix[pos_x, pos_y] == 1:
                            obs[3, pos_x, pos_y] = -1
                            break
                        # セルが角で真ん中に壁があるならbreak
                        if (
                            j == 0
                            and x != 0
                            and self.world.map.wall_matrix[pos_x - x, pos_y] == 1
                            and opr != x
                        ):
                            break
                        # 何もないなら0
                        else:
                            obs[3, pos_x, pos_y] = 0

        for y in range(-1, 2):
            for x in [-1, 1]:
                for opr in range(-1, 2):
                    for j in range(3):
                        pos_x, pos_y = x + j * x, y + j * opr
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
                            obs[3, pos_x, pos_y] = -1
                            break
                        # セルが真ん中で壁のない角の方向なら続ける
                        if (
                            j == 0
                            and y == 0
                            and self.world.map.wall_matrix[pos_x, pos_y - opr] == 0
                        ):
                            obs[3, pos_x, pos_y] = 0
                            continue
                        # 壁なら-1
                        if self.world.map.wall_matrix[pos_x, pos_y] == 1:
                            obs[3, pos_x, pos_y] = -1
                            break
                        # セルが角で真ん中に壁があるならbreak
                        if (
                            j == 0
                            and y != 0
                            and self.world.map.wall_matrix[pos_x, pos_y + y] == 1
                            and opr != y
                        ):
                            break
                        # 何もないなら0
                        else:
                            obs[3, pos_x, pos_y] = 0

        for opr_x in [-1, 1]:
            for x in range(self.visible_range // 2 + 1):
                # 壁ならbreak
                pos_x, pos_y = (
                    agent.x + (x * opr_x),
                    agent.y,
                )
                pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                if self.world.map.wall_matrix[pos_x, pos_y] == 1:
                    obs[3, pos_x, pos_y] = -1
                    break
                for opr_y in [-1, 1]:
                    for y in range(2):
                        pos_x, pos_y = (
                            agent.x + (x * opr_x),
                            agent.y + (y * opr_y),
                        )
                        pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                        # 場外なら-1
                        if (
                            pos_x < 0
                            or self.world.map.SIZE_X <= pos_x
                            or pos_y < 0
                            or self.world.map.SIZE_Y <= pos_y
                        ):
                            obs[3, pos_x, pos_y] = -1
                            continue
                        # 壁なら-1
                        if self.world.map.wall_matrix[pos_x, pos_y] == 1:
                            obs[3, pos_x, pos_y] = -1
                            break
                        # 何もないなら0
                        else:
                            obs[3, pos_x, pos_y] = 0

        for opr_y in [-1, 1]:
            for y in range(self.visible_range // 2 + 1):
                # 壁ならbreak
                pos_x, pos_y = agent.x, agent.y + (y * opr_y)
                pos_x, pos_y = self.world.map.coord2ind((pos_x, pos_y))
                if self.world.map.wall_matrix[pos_x, pos_y] == 1:
                    obs[3, pos_x, pos_y] = -1
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
                            obs[3, pos_x, pos_y] = -1
                            continue
                        # 壁なら-1
                        if self.world.map.wall_matrix[pos_x, pos_y] == 1:
                            obs[3, pos_x, pos_y] = -1
                            break
                        # 何もないなら0
                        else:
                            obs[3, pos_x, pos_y] = 0

        return obs

    def fill_obs_agent(self, obs, agent, offset_x, offset_y):
        obs[0, agent.x, agent.y] = 1
        for a in self.world.agents:
            diff_x, diff_y = a.xy - agent.xy
            if abs(diff_x) > 3 or abs(diff_y) > 3 or (diff_x == 0 and diff_y == 0):
                continue

            pos_x, pos_y = self.world.map.coord2ind(position=(a.x, a.y))
            # add if the object is within sight
            if obs[3, pos_x, pos_y] != -1:
                obs[1, pos_x, pos_y] = 1

        return obs

    def fill_obs_object(self, obs, agent, offset_x, offset_y):
        for obj in self.world.objects:
            diff_x, diff_y = obj.xy - agent.xy
            if abs(diff_x) > 3 or abs(diff_y) > 3:
                continue

            pos_x, pos_y = self.world.map.coord2ind(position=(obj.x, obj.y))
            # add if the object is within sight
            if obs[3, offset_x + pos_x, offset_y + pos_y] != -1:
                obs[2, offset_x + pos_x, offset_y + pos_y] = 1

        return obs
