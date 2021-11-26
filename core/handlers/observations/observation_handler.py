# -*- coding: utf-8 -*-

"""Source code for multi-agent observation handler method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from core.worlds import AbstractWorld
from core.worlds.entity import Agent
from omegaconf import DictConfig
from copy import deepcopy
from .agents import generate_observation_agent
from .masks import generate_observation_area_mask, generate_observation_mask_coordinate
from .noises import generate_observation_noise
from .objects import generate_observation_object


class ObservationHandler:
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
        self.view_method = config.observation_area_mask

        self.observation_area_mask = generate_observation_area_mask(
            config=config, world=world
        )
        self.observation_agent = generate_observation_agent(config=config, world=world)
        self.observation_object = generate_observation_object(
            config=config, world=world
        )
        self.observation_noise = generate_observation_noise(
            config=config, world=world, observation_space=self.observation_space
        )
        self.observation_mask_coordinate = generate_observation_mask_coordinate(
            config=config, world=world, observation_space=self.observation_space
        )

    @property
    def observation_space(self):
        agent_ch = self.observation_agent.get_channel()
        object_ch = self.observation_object.get_channel()

        return [agent_ch + object_ch + 1, *self.observation_area_mask.shape[2:]]

    def observation_ind(
        self, agents: List[Agent], agent: Agent, agent_id: int
    ) -> torch.Tensor:
        coordinates = self.observation_mask_coordinate.get_mask_coordinates(agent)

        # input walls and invisible area
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        area_mask = self.observation_area_mask[pos_x, pos_y]

        # input agents within sight
        obs_agent = self.observation_agent.fill(
            agents, agent, agent_id, area_mask, coordinates
        )

        # input objects within sight
        obs_object = self.observation_object.fill(
            agents, agent, agent_id, area_mask, coordinates
        )

        obs_area = self.observation_area_fill(
            agents, agent, agent_id, area_mask, coordinates
        )

        obs = torch.cat([obs_agent, obs_object, obs_area])

        # add observation noise
        obs = self.fill_obs_noise(obs, agent, agent_id)

        return {
            self.view_method: obs,
            "destination_channel": torch.from_numpy(
                deepcopy(self.world.map.destination_area_matrix[agent_id])
            ).unsqueeze(0),
        }

    def observation_area_fill(self, agents, agent, agent_id, area_mask, coordinates):
        return torch.from_numpy(area_mask).unsqueeze(0).float() - 1

    def fill_obs_noise(self, obs, agent, agent_id):
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        if self.world.map.noise_area_matrix[pos_x, pos_y] == 0:
            return obs

        obs = self.observation_noise.add_noise(obs, agent, agent_id)

        return obs

    def render(self, state: torch.Tensor) -> torch.Tensor:
        image = torch.zeros(3, *self.observation_space[1:])
        obs = state[self.view_method].permute(0, 2, 1)

        image, channel = self.observation_agent.render(obs, image, 0)
        image, channel = self.observation_object.render(obs, image, channel)
        # add invisible area information (White)
        image -= obs[channel]

        return {self.view_method: image}

    def reset(self, agents: List[Agent]) -> torch.Tensor:
        self.observation_agent.reset(agents, self.observation_mask_coordinate)
        self.observation_object.reset(agents, self.observation_mask_coordinate)

        obs_n = []
        for agent_id, agent in enumerate(agents):
            obs_n.append(self.observation_ind(agents, agent, agent_id))

        return obs_n

    def step(self, agents: List[Agent]):
        self.observation_agent.step(agents, self.observation_mask_coordinate)
        self.observation_object.step(agents, self.observation_mask_coordinate)
