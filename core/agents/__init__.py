# -*- coding: utf-8 -*-

"""Builds agents that learn the environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

from core.utils.logging import initialize_logging
from omegaconf import DictConfig

from .abstract_agent import AbstractAgent
from .default_agent import DefaultAgent
from .mat_agent import MATAgent

logger = initialize_logging(__name__)

__all__ = ["AbstractAgent", "DefaultAgent", "MATAgent"]


def generate_agents(
    config: DictConfig, observation_space: List[List[int]], action_space: List[int]
) -> AbstractAgent:
    if config.agent == "default":
        agents = [
            DefaultAgent(config, obs_shape, act_size)
            for obs_shape, act_size in zip(observation_space, action_space)
        ]

    elif config.agent == "mat":
        agents = [
            MATAgent(config, obs_shape, act_size)
            for obs_shape, act_size in zip(observation_space, action_space)
        ]

    else:
        logger.warn(f"Unexpected agent is given. config.agent: {config.agent}")

        raise ValueError()

    return agents
