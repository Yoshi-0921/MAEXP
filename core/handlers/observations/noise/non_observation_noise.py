# -*- coding: utf-8 -*-

"""Source code for non observation noise.
That is, no noise is added in the observation phase.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch

from .abstract_observation_noise import AbstractObservationNoise


class NonObservationNoise(AbstractObservationNoise):
    def get_noise(self, agent, agent_id, offset_x, offset_y):
        noise = torch.zeros(self.observation_space)

        return noise
