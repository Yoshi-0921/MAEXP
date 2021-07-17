# -*- coding: utf-8 -*-

"""Source code for simple environmental map used in multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from omegaconf import DictConfig

from .abstract_map import AbstractMap


class SimpleMap(AbstractMap):
    def __init__(self, config: DictConfig):
        super().__init__(
            config=config, size_x=config.map.SIZE_X, size_y=config.map.SIZE_Y
        )
