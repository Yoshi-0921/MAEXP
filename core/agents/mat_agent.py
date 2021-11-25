# -*- coding: utf-8 -*-

"""Source code for mat agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from random import random

import torch

from .abstract_agent import AbstractAgent


class MATAgent(AbstractAgent):
    def get_action_attns(self, state, epsilon):
        action, attns = self.brain.get_action(state)

        # attns = torch.stack(attns)
        attns = [torch.stack(attn) for attn in attns]

        if random() < epsilon:
            action = self.get_random_action()

        return action, attns

    def get_action(self, state, epsilon):
        action, _ = self.brain.get_action(state)

        if random() < epsilon:
            action = self.get_random_action()

        return action
