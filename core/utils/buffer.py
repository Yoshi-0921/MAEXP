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
            final = [self.buffer[i] for i in range(max(start + 1, 0), end + 1)]

            # correct for sampling across episodes
            for i in range(len(final) - 2, -1, -1):
                if final[i][3][0]:
                    final = final[i + 1:]
                    break

            # pad beginning to account for corrections
            while len(final) < self.seq_length:
                final = Experience() + final

            sampled_exp.append(final)

        return tuple(zip(sampled_exp))
