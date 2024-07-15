# -*- coding: utf-8 -*-

"""Source code for flip observation noise that varies depending on distance from center.
Furthrer distance, more likely to flip in the observation.
Noise occurs independently at each channel.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig

from core.worlds.abstract_world import AbstractWorld

from .abstract_observation_noise import AbstractObservationNoise


class FlipObservatonNoise(AbstractObservationNoise):
    def __init__(
        self, config: DictConfig, world: AbstractWorld, observation_space: List[int]
    ):
        super().__init__(
            config=config, world=world, observation_space=observation_space
        )
        self.flip_noise_probabilities = config.flip_noise_probabilities
        self.noise_range = 0.5 / (torch.tensor(self.flip_noise_probabilities) - 1)

    def add_noise(self, obs, agent, agent_id):
        noised_obs = torch.zeros(self.observation_space)
        _, x, y = self.observation_space
        obs[2, ...] += 1

        for i in range(min(x, y) // 2):
            noised_obs[:, i:x, i:y] = obs[:, i:x, i:y] + (
                torch.rand((self.observation_space[0], (x - i), (y - i)))
                * self.noise_range[i]
            )
            x -= 1
            y -= 1

        noised_obs = torch.abs(torch.round(noised_obs))
        noised_obs[2, ...] -= 1

        return noised_obs
