# -*- coding: utf-8 -*-

"""Source code for dqn agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from random import random

from omegaconf import DictConfig

from .abstract_agent import AbstractAgent
from .brains import generate_brain
from typing import List


class DefaultAgent(AbstractAgent):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(obs_shape=obs_shape, act_size=act_size, config=config)
        self.brain = generate_brain(config=config, obs_shape=obs_shape, act_size=act_size)

    def get_action(self, state, epsilon):
        if random() < epsilon:
            action = self.get_random_action()

        else:
            action = self.brain.get_action(state)

        return action

    def learn(self, state, action, reward, done, next_state):
        loss = self.brain.learn(state, action, reward, done, next_state)

        return loss
