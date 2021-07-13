# -*- coding: utf-8 -*-

"""Source code for observation noise that varies depending on distance from center.
Furthrer distance, more likely to add noise in the observation.
Noise is added to each channel.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig
import torch

from .abstract_observation_noise import AbstractObservationNoise


class DistObservatonNoise(AbstractObservationNoise):
    def __init__(self, config: DictConfig, world: AbstractWorld, observation_space: List[int]):
        super().__init__(config=config, world=world, observation_space=observation_space)
        self.std_distribution = config.std_distribution

    def add_noise(self, obs, agent, agent_id, offset_x, offset_y):
        noise = torch.zeros(self.observation_space)
        _, x, y = self.observation_space
        for i in range(min(x, y) // 2):
            noise[:, i:x, i:y] = torch.empty((self.observation_space[0], (x - i), (y - i))).normal_(mean=0, std=self.std_distribution[i])
            x -= 1
            y -= 1

        obs += noise

        return obs


class ThresholdDistObservationNoise(DistObservatonNoise):
    def __init__(self, config: DictConfig, world: AbstractWorld, observation_space: List[int]):
        super().__init__(config=config, world=world, observation_space=observation_space)
        self.threshold_ratio = config.noise_threshold_ratio

    def add_noise(self, obs, agent, agent_id, offset_x, offset_y):
        noise = torch.zeros(self.observation_space)
        _, x, y = self.observation_space
        for i in range(min(x, y) // 2):
            generated_noise = torch.empty((self.observation_space[0], (x - i), (y - i))).normal_(mean=0, std=self.std_distribution[i])
            noise[:, i:x, i:y] = torch.where(
                generated_noise < -self.std_distribution[i] * self.threshold_ratio,
                generated_noise, noise[:, i:x, i:y]
            )
            noise[:, i:x, i:y] = torch.where(
                self.std_distribution[i] * self.threshold_ratio < generated_noise,
                generated_noise, noise[:, i:x, i:y]
            )
            x -= 1
            y -= 1

        obs += noise

        return obs
