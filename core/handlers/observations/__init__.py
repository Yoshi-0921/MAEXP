# -*- coding: utf-8 -*-

"""Builds observation handler used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from core.worlds.abstract_world import AbstractWorld
from omegaconf import DictConfig

from .abstract_observation import AbstractObservation
from .local_view_observation import LocalViewObservaton
from .relative_view_observation import RelativeViewObservaton

logger = initialize_logging(__name__)

__all__ = ["AbstractObservation", "LocalViewObservation", "RelativeViewObservation"]


def generate_observation_handler(
    config: DictConfig, world: AbstractWorld
) -> AbstractObservation:
    if config.view_method == "local_view":
        obs = LocalViewObservaton(config=config, world=world)

    elif config.view_method == "relative_view":
        obs = RelativeViewObservaton(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected view_method is given. config.view_method: {config.view_method}"
        )

        raise ValueError()

    return obs