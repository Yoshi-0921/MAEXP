# -*- coding: utf-8 -*-

"""Source code for the default multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.maps.abstract_map import AbstractMap
from omegaconf import DictConfig

from .abstract_world import AbstractWorld
from .entity import Agent


class DefaultWorld(AbstractWorld):
    def __init__(self, config: DictConfig, world_map: AbstractMap):
        super().__init__(config=config, world_map=world_map)

        self.agents = [Agent(name=f"Agent_{i}") for i in range(config.num_agents)]
