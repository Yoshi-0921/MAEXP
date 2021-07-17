# -*- coding: utf-8 -*-

"""Source code for observation handler using the local view method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch
from core.worlds.entity import Agent
from typing import List

from .abstract_observation import AbstractObservation


class LocalViewObservaton(AbstractObservation):
    @property
    def observation_space(self) -> List[int]:
        # 0:agents, 1:objects, 2:visible area
        return [3, self.visible_range, self.visible_range]

    def observation_ind(self, agent: Agent, agent_id: int) -> torch.Tensor:
        obs = torch.zeros(self.observation_space)
        offset = 0

        # input walls and invisible area
        obs = self.fill_obs_area(obs, agent, agent_id, offset, offset)

        # input objects within sight
        obs = self.fill_obs_object(obs, agent, agent_id, offset, offset)

        # input agents within sight
        obs = self.fill_obs_agent(obs, agent, agent_id, offset, offset)

        # add observation noise
        obs = self.fill_obs_noise(obs, agent, agent_id, offset, offset)

        return obs

    def fill_obs_area(self, obs, agent, agent_id, offset_x, offset_y) -> torch.Tensor:
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        obs[2, :, :] = self.observation_area_mask[pos_x, pos_y]

        return obs

    def fill_obs_agent(self, obs, agent, agent_id, offset_x, offset_y) -> torch.Tensor:
        obs[0, self.visible_radius, self.visible_radius] = 1
        for a in self.world.agents:
            diff_x, diff_y = a.xy - agent.xy
            if (
                abs(diff_x) > self.visible_radius
                or abs(diff_y) > self.visible_radius
                or (diff_x == 0 and diff_y == 0)
            ):
                continue

            pos_x, pos_y = self.world.map.coord2ind(
                position=(a.x - agent.x, a.y - agent.y),
                size_x=self.visible_range,
                size_y=self.visible_range,
            )
            # add if the object is within sight
            if obs[2, offset_x + pos_x, offset_y + pos_y] != -1:
                obs[0, offset_x + pos_x, offset_y + pos_y] = 1

        return obs

    def fill_obs_object(self, obs, agent, agent_id, offset_x, offset_y) -> torch.Tensor:
        for obj in self.world.objects:
            diff_x, diff_y = obj.xy - agent.xy
            if abs(diff_x) > self.visible_radius or abs(diff_y) > self.visible_radius:
                continue

            pos_x, pos_y = self.world.map.coord2ind(
                position=(obj.x - agent.x, obj.y - agent.y),
                size_x=self.visible_range,
                size_y=self.visible_range,
            )
            # add if the object is within sight
            if obs[2, offset_x + pos_x, offset_y + pos_y] != -1:
                obs[1, offset_x + pos_x, offset_y + pos_y] = 1

        return obs

    def fill_obs_noise(self, obs, agent, agent_id, offset_x, offset_y) -> torch.Tensor:
        return self.observation_noise.add_noise(
            obs, agent, agent_id, offset_x, offset_y
        )

    def render(self, state) -> torch.Tensor:
        image = torch.zeros(self.observation_space)
        obs = state.permute(0, 2, 1)

        # add agent information (Blue)
        image[2] += obs[0]
        # add object information (Yellow)
        image[torch.tensor([0, 1])] += obs[1]
        # add invisible area information (White)
        image -= obs[2]

        return image
