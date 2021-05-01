# -*- coding: utf-8 -*-

"""Source code for abstract agent class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC, abstractmethod
from random import random


class AbstractAgent(ABC):
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    def random_action(self) -> int:
        """
        Returns:
            random action
        """
        action = int(random() * 4)

        return action

    @abstractmethod
    def get_action(self, state, epsilon):
        raise NotImplementedError()
