# -*- coding: utf-8 -*-

"""Source code for replay buffer used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import collections
from typing import Tuple

import numpy as np
import torch

Experience = collections.namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'new_state']
)


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int, action_onehot=False, state_conv=False) -> None:
        self.buffer = collections.deque(maxlen=capacity)
        self.action_onehot = action_onehot
        self.state_conv = state_conv

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        global_states, global_actions, global_rewards, global_dones, global_next_states = zip(*[self.buffer[idx] for idx in indices])

        if self.state_conv:
            global_states = torch.stack(global_states).permute(1, 0, 2, 3, 4)
        else:
            global_states = torch.stack(global_states).permute(1, 0, 2)
        global_rewards = torch.tensor(global_rewards).permute(1, 0).float()
        global_dones = torch.tensor(global_dones).permute(1, 0)
        if self.state_conv:
            global_next_states = torch.stack(global_next_states).permute(1, 0, 2, 3, 4)
        else:
            global_next_states = torch.stack(global_next_states).permute(1, 0, 2)

        if self.action_onehot:
            global_actions = torch.tensor(global_actions).permute(1, 0, 2)
        else:
            global_actions = torch.tensor(global_actions).permute(1, 0)

        return global_states, global_actions, global_rewards, global_dones, global_next_states
