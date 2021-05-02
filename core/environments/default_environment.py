# -*- coding: utf-8 -*-

"""Source code for default multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from .abstract_environment import AbstractEnvironment
from omegaconf import DictConfig


class DefaultEnvironment(AbstractEnvironment):
    def __init__(self, config: DictConfig, world):
        super().__init__(config=config, world=world)
