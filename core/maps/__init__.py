# -*- coding: utf-8 -*-

"""Source code to generate multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_map import AbstractMap
from .four_rectangle import FourRectangleMap
from .four_rooms_map import FourRoomsMap
from .simple_map import SimpleMap
from .three_rooms_map import ThreeRoomsMap

logger = initialize_logging(__name__)


def generate_map(config: DictConfig) -> AbstractMap:
    if config.map.name == "simple":
        world_map = SimpleMap(config=config)

    elif config.map.name == "four_rooms":
        world_map = FourRoomsMap(config=config)

    elif config.map.name == "three_rooms":
        world_map = ThreeRoomsMap(config=config)

    elif config.map.name == "four_rectangle":
        world_map = FourRectangleMap(config=config)

    else:
        logger.warn(f"Unexpected map is given. config.map.name: {config.map.name}")

        raise ValueError()

    return world_map
