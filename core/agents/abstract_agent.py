# -*- coding: utf-8 -*-

"""Source code for abstract agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from random import random


class AbstractAgent(ABC):
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, config, obs_shape, act_size) -> None:
        self.config = config
        self.obs_shape = obs_shape
        self.act_size = act_size
        self.brain = None

    def get_random_action(self) -> int:
        action = int(random() * self.act_size)

        return action

    @abstractmethod
    def get_action(self, state, epsilon):
        raise NotImplementedError()

    @abstractmethod
    def learn(self, state, action, reward, done, next_state):
        raise NotImplementedError()
