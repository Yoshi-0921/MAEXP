# -*- coding: utf-8 -*-

"""Builds agents that learn the environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_agent import AbstractAgent
from .default_agent import DefaultAgent

logger = initialize_logging(__name__)

__all__ = ["AbstractAgent", "DefaultAgent"]


def generate_agents(
    config: DictConfig, observation_space: List[int], action_space: List[int]
) -> AbstractAgent:
    if config.agents == "default":
        agents = [
            DefaultAgent(config, obs_size, act_size)
            for obs_size, act_size in zip(observation_space, action_space)
        ]

        return agents

    else:
        logger.warn(f"Unexpected agent is given. config.agent: {config.agent}")

        raise ValueError()
