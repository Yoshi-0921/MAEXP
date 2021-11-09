# -*- coding: utf-8 -*-

"""Source code for transition observation handler for agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import numpy as np
import torch
from core.utils.color import RGB_COLORS
from core.worlds import AbstractWorld
from omegaconf import DictConfig

from ..abstract_observation_handler import AbstractObservationHandler


class TransitionAgentObservationHandler(AbstractObservationHandler):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.num_agents = config.num_agents
        self.past_step = config.past_step
        assert 0 < self.past_step
        super().__init__(config=config, world=world)

    def get_channel(self):
        # 0: agents(t), 1: agents(t-1), 2: agents(t-2), ..., n: agents(t-n)
        return 1 + self.past_step

    def fill(self, agents, agent, agent_id, area_mask, coordinates):
        obs = torch.zeros(self.get_channel(), *area_mask.shape)
        obs[
            0,
            coordinates["obs_x_min"]: coordinates["obs_x_max"],
            coordinates["obs_y_min"]: coordinates["obs_y_max"],
        ] = torch.from_numpy(
            area_mask[
                coordinates["area_x_min"]: coordinates["area_x_max"],
                coordinates["area_y_min"]: coordinates["area_y_max"],
            ]
            * self.world.map.agents_matrix[
                :,
                coordinates["map_x_min"]: coordinates["map_x_max"],
                coordinates["map_y_min"]: coordinates["map_y_max"],
            ].sum(axis=0)
        )

        for t in range(self.past_step):
            obs[
                t + 1,
                coordinates["obs_x_min"]: coordinates["obs_x_max"],
                coordinates["obs_y_min"]: coordinates["obs_y_max"],
            ] += torch.from_numpy(
                self.past_global_agents[t][agent_id][
                    coordinates["map_x_min"]: coordinates["map_x_max"],
                    coordinates["map_y_min"]: coordinates["map_y_max"],
                ]
            )

        return obs

    def render(self, obs, image, channel):
        # add agent information (Blue)
        rgb = RGB_COLORS["blue"]
        rgb = np.expand_dims(np.asarray(rgb), axis=(1, 2))
        image += obs[channel] * rgb
        for t in range(1, self.past_step + 1):
            image += obs[channel + t] * rgb * (0.5 ** t)

        return torch.clamp(image, min=0., max=1.), channel + self.get_channel()

    def step(self, agents, coordinate_handler):
        # past_global_agents: [[t-1], [t-2], ..., [t-n]]
        self.past_global_agents.pop()
        self.past_global_agents.insert(0, self.get_past_global_agents(agents, coordinate_handler))

    def reset(self, agents, coordinate_handler):
        self.past_global_agents = [
            self.get_past_global_agents(agents, coordinate_handler) for _ in range(self.past_step)
        ]

    def get_past_global_agents(self, agents, coordinate_handler):
        global_agents = [
            self.world.map.agents_matrix.sum(axis=0).copy() for _ in range(self.num_agents)
        ]
        for agent_id, agent in enumerate(agents):
            coordinates = coordinate_handler.get_mask_coordinates(agent)
            mask = np.zeros_like(global_agents[0])
            mask[
                coordinates["map_x_min"]: coordinates["map_x_max"],
                coordinates["map_y_min"]: coordinates["map_y_max"],
            ] += 1
            global_agents[agent_id] *= mask

        return global_agents
