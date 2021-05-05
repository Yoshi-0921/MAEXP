
from random import random

from .abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, obs_size, act_size, config):
        super().__init__(obs_size=obs_size, act_size=act_size, config=config)

    def get_action(self, state, epsilon):
        action = int(random() * self.act_size)

        return action

    def update(self, state, action, reward, done, next_state):
        pass
