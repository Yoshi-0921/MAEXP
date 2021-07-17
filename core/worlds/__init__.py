# -*- coding: utf-8 -*-

"""Source code to generate multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.maps.abstract_map import AbstractMap
from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_world import AbstractWorld
from .default_world import DefaultWorld

logger = initialize_logging(__name__)


def generate_world(config: DictConfig, world_map: AbstractMap) -> AbstractWorld:
    if config.world == "default":
        world = DefaultWorld(config=config, world_map=world_map)

        return world

    else:
        logger.warn(f"Unexpected world is given. config.world: {config.world}")

        raise ValueError()
