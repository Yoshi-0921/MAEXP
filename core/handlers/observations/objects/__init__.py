# -*- coding: utf-8 -*-

"""Builds object observation handler used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from core.worlds import AbstractWorld

from .default_object_observation import DefaultObjectObservationHandler
from .simple_object_observation import SimpleObjectObservationHandler

logger = initialize_logging(__name__)


def generate_observation_object(config, world: AbstractWorld):
    if config.object_view_method == "default":
        observation_object = DefaultObjectObservationHandler(config=config, world=world)

    elif config.object_view_method == "simple":
        observation_object = SimpleObjectObservationHandler(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected object_view_method is given. config.object_view_method: {config.object_view_method}"
        )

        raise ValueError()

    return observation_object
