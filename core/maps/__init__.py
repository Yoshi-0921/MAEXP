# -*- coding: utf-8 -*-

"""Source code to generate multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_map import AbstractMap
from .default_map import DefaultMap

logger = initialize_logging(__name__)


def generate_map(config: DictConfig) -> AbstractMap:
    if config.map == 'default':
        world_map = DefaultMap(config=config)

        return world_map

    else:
        logger.warn(f"Unexpected map is given. config.map: {config.map}")

        raise ValueError()
