# -*- coding: utf-8 -*-

"""Source code for Entity object.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC
import numpy as np


class Entity(ABC):
    def __init__(self, name: str = None):
        self.__name = name
        self.__xy = np.zeros(2, dtype=np.int8)

    @property
    def name(self):
        return self.__name

    @property
    def x(self):
        return self.__xy[0]

    @property
    def y(self):
        return self.__xy[1]

    @property
    def xy(self):
        return self.__xy

    def push(self, movement: np.ndarray):
        self.__xy += movement

    def move(self, movement: np.ndarray):
        self.__xy = movement


class Object(Entity):
    def __init__(self):
        super().__init__()


class Agent(Entity):
    def __init__(self, name: str = None):
        super().__init__(name=name)
        self.collide_walls: bool = False
        self.collide_agents: bool = False
        self.action = np.zeros(2, dtype=np.int8)
