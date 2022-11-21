"""Source code for recurrent attention agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from random import random

import torch

from .abstract_agent import AbstractAgent


class RecurrentAttentionAgent(AbstractAgent):
    def get_action_attns(self, state, epsilon, hidden_vector=None):
        action, attns, hidden_vector = self.brain.get_action(state, hidden_vector)

        attns = [torch.stack(attn) for attn in attns]

        if random() < epsilon:
            action = self.get_random_action()

        return action, attns, hidden_vector

    def get_action(self, state, epsilon, hidden_vector=None):
        action, _, hidden_vector = self.get_action_attns(state=state, epsilon=epsilon, hidden_vector=hidden_vector)

        return action, hidden_vector
