"""Source code for abstract agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from random import random
from typing import List

from omegaconf import DictConfig

from .brains import generate_brain


class AbstractAgent(ABC):
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int) -> None:
        self.config = config
        self.obs_shape = obs_shape
        self.act_size = act_size
        self.brain = generate_brain(
            config=config, obs_shape=obs_shape, act_size=act_size
        )

    def get_random_action(self) -> int:
        action = int(random() * self.act_size)

        return action

    def synchronize_brain(self):
        self.brain.synchronize_network()

    def learn(self, state, action, reward, done, next_state):
        loss = self.brain.learn(state, action, reward, done, next_state)

        return loss

    @abstractmethod
    def get_action(self, state, epsilon):
        raise NotImplementedError()
