"""Source code for replay buffer used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import collections
from typing import Tuple

import numpy as np
from core.utils.logging import initialize_logging
from omegaconf import DictConfig

logger = initialize_logging(__name__)

Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


def generate_buffer(config: DictConfig):
    if config.buffer == "default":
        buffer = ReplayBuffer(capacity=config.capacity)

    elif config.buffer == "recurrent":
        buffer = RecurrentReplayBuffer(capacity=config.capacity, sequence_length=config.sequence_length)

    else:
        logger.warn(f"Unexpected buffer is given. config.buffer: {config.buffer}")

        raise ValueError()

    return buffer


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(
        self,
        capacity: int,
    ) -> None:
        self.buffer = collections.deque(maxlen=capacity)

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

        return tuple(zip(*[self.buffer[idx] for idx in indices]))


class RecurrentReplayBuffer(ReplayBuffer):
    """https://github.com/qfettes/DeepRL-Tutorials/blob/master/11.DRQN.ipynb"""

    def __init__(self, capacity, sequence_length=10):
        super().__init__(capacity=capacity)
        self.seq_length = sequence_length

    def sample(self, batch_size: int) -> Tuple:
        end_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        start_indices = end_indices - self.seq_length
        sampled_exp = []

        for start, end in zip(start_indices, end_indices):
            # correct for sampling near beginning
            sequence_exp = list(map(list, zip(*[self.buffer[i] for i in range(max(start + 1, 0), end + 1)])))
            final = self.transpose(sequence_exp)

            sampled_exp.append(final)

        return tuple(zip(*sampled_exp))

    def transpose(self, sequence_exp):
        # correct for sampling across episodes
        states, actions, rewards, dones, new_states = sequence_exp
        sequence_len = 0
        for sequence_len in range(len(dones) - 2, -1, -1):
            if dones[sequence_len][0]:
                sequence_len += 1
                break

        # pad beginning to account for corrections
        padding_len = self.seq_length - len(dones)
        states = [states[sequence_len] for _ in range(sequence_len + padding_len)] + states[sequence_len:]
        actions = [actions[sequence_len] for _ in range(sequence_len + padding_len)] + actions[sequence_len:]
        rewards = [rewards[sequence_len] for _ in range(sequence_len + padding_len)] + rewards[sequence_len:]
        dones = [dones[sequence_len] for _ in range(sequence_len + padding_len)] + dones[sequence_len:]
        new_states = [new_states[sequence_len] for _ in range(sequence_len + padding_len)] + new_states[sequence_len:]

        transposed_states = tuple(zip(*states))
        states_actions = tuple(zip(*actions))
        states_rewards = tuple(zip(*rewards))
        states_dones = tuple(zip(*dones))
        states_new_states = tuple(zip(*new_states))

        return (transposed_states, states_actions, states_rewards, states_dones, states_new_states)
