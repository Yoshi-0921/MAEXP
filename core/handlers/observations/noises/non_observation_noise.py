# -*- coding: utf-8 -*-

"""Source code for non observation noise.
That is, no noise is added in the observation phase.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from .abstract_observation_noise import AbstractObservationNoise


class NonObservationNoise(AbstractObservationNoise):
    def add_noise(self, obs, agent, agent_id):
        return obs
