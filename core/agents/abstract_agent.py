# -*- coding: utf-8 -*-

"""Source code for abstract agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
import torch


class AbstractAgent(ABC):
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, obs_size, act_size, config) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dqn = None
        self.dqn_target = None
        self.criterion = None

    @abstractmethod
    def get_action(self, state, epsilon):
        raise NotImplementedError()

    @abstractmethod
    def update(self, state, action, reward, done, next_state):
        raise NotImplementedError()
