# -*- coding: utf-8 -*-

"""Source code for random agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from omegaconf import DictConfig

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, config: DictConfig, obs_size: int, act_size: int):
        super().__init__(obs_size=obs_size, act_size=act_size, config=config)

    def get_action(self, state, epsilon):

        return super().get_random_action()

    def learn(self, state, action, reward, done, next_state):
        pass

    def synchronize_brain(self):
        pass
