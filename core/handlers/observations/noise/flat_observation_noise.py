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


class FlatObservatonNoise(AbstractObservationNoise):
    def __init__(self, config: DictConfig, world: AbstractWorld, observation_space: List[int]):
        super().__init__(config=config, world=world, observation_space=observation_space)

    def get_noise(self, agent, agent_id, offset_x, offset_y):
        _, x, y = self.observation_space
        noise = torch.empty((3, x, y)).normal_(mean=0, std=0.03)

        return noise
