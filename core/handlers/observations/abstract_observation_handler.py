from abc import ABC, abstractmethod

import torch
from core.worlds import AbstractWorld
from omegaconf import DictConfig


class AbstractObservationHandler(ABC):
    def __init__(
        self, config: DictConfig, world: AbstractWorld
    ):
        self.config = config
        self.world = world
        self.visible_range = config.visible_range
        self.visible_radius = config.visible_range // 2

    @abstractmethod
    def get_channel(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def fill(self, agents, agent, agent_id, area_mask, coordinates) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def render(self, obs, image, channel) -> torch.Tensor:
        raise NotImplementedError()

    def step(self, agents, coordinate_handler):
        pass

    def reset(self, agents, coordinate_handler):
        pass
