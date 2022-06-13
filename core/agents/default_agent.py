"""Source code for dqn agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from random import random

from .abstract_agent import AbstractAgent


class DefaultAgent(AbstractAgent):
    def get_action(self, state, epsilon):
        if random() < epsilon:
            action = self.get_random_action()

        else:
            action = self.brain.get_action(state)

        return action
