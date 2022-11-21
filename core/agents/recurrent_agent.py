"""Source code for recurrent dqn agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from random import random

from .default_agent import DefaultAgent


class RecurrentAgent(DefaultAgent):
    def get_action(self, state, epsilon, hidden_vector=None):
        if random() < epsilon:
            action = self.get_random_action()

        else:
            action, hidden_vector = self.brain.get_action(state, hidden_vector)

        return action, hidden_vector