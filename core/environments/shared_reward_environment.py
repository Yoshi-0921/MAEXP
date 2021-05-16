# -*- coding: utf-8 -*-

"""Source code for default multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import numpy as np

from .default_environment import DefaultEnvironment


class SharedMeanRewardEnvironment(DefaultEnvironment):
    def step(self, action_n: List[np.array]):
        reward_n, done_n, obs_n = super().step(action_n)
        shared_reward_n = [np.asarray(reward_n).mean() for _ in range(self.num_agents)]

        return shared_reward_n, done_n, obs_n


class SharedMaxRewardEnvironment(DefaultEnvironment):
    def step(self, action_n: List[np.array]):
        reward_n, done_n, obs_n = super().step(action_n)
        shared_reward_n = [np.asarray(reward_n).mean() for _ in range(self.num_agents)]

        return shared_reward_n, done_n, obs_n
