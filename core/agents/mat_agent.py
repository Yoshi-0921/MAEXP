# -*- coding: utf-8 -*-

"""Source code for mat agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from random import random

from omegaconf import DictConfig

from .abstract_agent import AbstractAgent
from .brains import generate_brain
from typing import List
import torch


class MATAgent(AbstractAgent):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(obs_shape=obs_shape, act_size=act_size, config=config)
        self.brain = generate_brain(
            config=config, obs_shape=obs_shape, act_size=act_size
        )
        self.patched_size_x = self.brain.network.patched_size_x
        self.patched_size_y = self.brain.network.patched_size_y

        attn_shape = [
            config.model.num_heads,
            self.patched_size_x * self.patched_size_y + 1,
            self.patched_size_x * self.patched_size_y + 1,
        ]
        self.attn_shape = [torch.zeros([1, *attn_shape]) for _ in range(self.config.model.block_loop)]

    def get_action(self, state, epsilon):
        if random() < epsilon:
            action = self.get_random_action()
            attns = self.attn_shape

        else:
            action, attns = self.brain.get_action(state)

        attns = torch.stack(attns)
        return action, attns

    def learn(self, state, action, reward, done, next_state):
        loss = self.brain.learn(state, action, reward, done, next_state)

        return loss