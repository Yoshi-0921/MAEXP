"""Source code for multi-agent observation handler method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from copy import deepcopy
from typing import Dict, List, Union

import torch
from omegaconf import DictConfig

from core.utils.logging import initialize_logging
from core.worlds import AbstractWorld
from core.worlds.entity import Agent

from .agents import generate_observation_agent
from .masks import (generate_observation_area_mask,
                    generate_observation_mask_coordinate)
from .noises import generate_observation_noise
from .objects import generate_observation_object

logger = initialize_logging(__name__)


class ObservationHandler:
    def __init__(self, config: DictConfig, world: AbstractWorld):
        self.config = config
        self.world = world
        self.view_method = config.observation_area_mask

        if self.view_method == "local":
            self.observation_size = [config.visible_range, config.visible_range]
        elif self.view_method == "relative":
            self.observation_size = [world.map.SIZE_X, world.map.SIZE_Y]
        else:
            logger.warn(
                f"Unexpected view_method is given. self.view_method: {self.view_method}"
            )
            raise ValueError()

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
            config=config, world=world
        )

    @property
    def observation_space(self):
        agent_ch = self.observation_agent.get_channel()
        object_ch = self.observation_object.get_channel()

        return [agent_ch + object_ch + 1, *self.observation_size]

    def observation_ind(
        self, agents: List[Agent], agent: Agent, agent_id: int
    ) -> Dict[str, Union[torch.Tensor, int]]:
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
            "coordinates": coordinates,
            "agents": torch.stack([torch.from_numpy(self.world.map.coord2ind(agent_ind.xy)) for agent_ind in agents]),
            "objects": torch.from_numpy(
                deepcopy(self.world.map.objects_matrix)
            ),
            "destination": torch.from_numpy(
                deepcopy(self.world.map.destination_area_matrix[agent_id, :, :])
            )
        }

    def observation_area_fill(self, agents, agent, agent_id, area_mask, coordinates):
        return torch.from_numpy(area_mask).unsqueeze(0).float() - 1

    def fill_obs_noise(self, obs, agent, agent_id):
        pos_x, pos_y = self.world.map.coord2ind(agent.xy)
        if self.world.map.noise_area_matrix[pos_x, pos_y] == 0:
            return obs

        obs = self.observation_noise.add_noise(obs, agent, agent_id)

        return obs

    def render(self, state: Dict[str, Union[torch.Tensor, int]]) -> torch.Tensor:
        image = torch.zeros(3, *self.observation_size)

        if self.view_method == "local":
            obs = state[self.view_method]
        elif self.view_method == "relative":
            obs = ObservationHandler.decode_relative_state(
                state=state, observation_size=self.observation_size
            )

        image, channel = self.observation_agent.render(obs, image, 0)
        image, channel = self.observation_object.render(obs, image, channel)
        # add invisible area information (White)
        image -= obs[channel]

        return {self.view_method: image.permute(0, 2, 1)}

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

    @staticmethod
    def decode_relative_state(
        state: Dict[str, Union[torch.Tensor, int]], observation_size: List[int]
    ) -> torch.Tensor:
        assert "relative" in state.keys()

        if len(state["relative"].shape) == 3:
            decoded_state = torch.zeros(
                size=[state["relative"].shape[0], *observation_size]
            )
            decoded_state[-1, ...] -= 1
            coordinates = deepcopy(state["coordinates"])
            decoded_state[
                :,
                coordinates["map_x_min"]: coordinates["map_x_max"],
                coordinates["map_y_min"]: coordinates["map_y_max"],
            ] = state["relative"][
                :,
                coordinates["obs_x_min"]: coordinates["obs_x_max"],
                coordinates["obs_y_min"]: coordinates["obs_y_max"],
            ]

        elif len(state["relative"].shape) == 4:
            decoded_state = torch.zeros(
                size=[*state["relative"].shape[:2], *observation_size],
                device=state["relative"].device,
            )
            decoded_state[:, -1, ...] -= 1
            coordinates = deepcopy(state["coordinates"])
            if decoded_state.shape[0] == 1:
                for coordinate_key, coordinate_value in coordinates.items():
                    coordinates[coordinate_key] = [coordinate_value]

            for batch_id, (
                map_x_min,
                map_x_max,
                map_y_min,
                map_y_max,
                obs_x_min,
                obs_x_max,
                obs_y_min,
                obs_y_max,
            ) in enumerate(
                zip(
                    coordinates["map_x_min"],
                    coordinates["map_x_max"],
                    coordinates["map_y_min"],
                    coordinates["map_y_max"],
                    coordinates["obs_x_min"],
                    coordinates["obs_x_max"],
                    coordinates["obs_y_min"],
                    coordinates["obs_y_max"],
                )
            ):
                decoded_state[
                    batch_id, :, map_x_min:map_x_max, map_y_min:map_y_max
                ] = state["relative"][
                    batch_id, :, obs_x_min:obs_x_max, obs_y_min:obs_y_max
                ]

        else:
            logger.warn(
                f"Unexpected state length is given. len(state['relative'].shape): {len(state['relative'].shape)}"
            )
            raise ValueError()

        return decoded_state

    @staticmethod
    def decode_agents_channel(
        state: Dict[str, Union[torch.Tensor, int]], observation_size: List[int]
    ) -> torch.Tensor:
        assert "agents" in state.keys()

        if len(state["agents"].shape) == 2:
            decoded_state = torch.zeros(
                size=[state["agents"].shape[0], *observation_size],
                device=state["agents"].device
            )
            for agent_id, coordinate in enumerate(state["agents"]):
                decoded_state[agent_id, coordinate[0].int(), coordinate[1].int()] = 1

        elif len(state["agents"].shape) == 3:
            decoded_state = torch.zeros(
                size=[*state["agents"].shape[:2], *observation_size],
                device=state["agents"].device,
            )

            for batch_id, coordinates in enumerate(state["agents"]):
                for agent_id, coordinate in enumerate(coordinates):
                    decoded_state[
                        batch_id, agent_id, coordinate[0].int(), coordinate[1].int()
                    ] = 1

        else:
            logger.warn(
                f"Unexpected state length is given. len(state['agents'].shape): {len(state['agents'].shape)}"
            )
            raise ValueError()

        return decoded_state
