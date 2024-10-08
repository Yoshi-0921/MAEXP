"""Builds agents that learn the environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

from omegaconf import DictConfig

from core.utils.logging import initialize_logging

from .abstract_agent import AbstractAgent
from .attention_agent import AttentionAgent
from .default_agent import DefaultAgent
from .recurrent_agent import RecurrentAgent
from .recurrent_attention_agent import RecurrentAttentionAgent

logger = initialize_logging(__name__)

__all__ = ["AbstractAgent", "DefaultAgent", "AttentionAgent", "RecurrentAgent", "RecurrentAttentionAgent"]


def generate_agents(
    config: DictConfig, observation_space: List[List[int]], action_space: List[int]
) -> AbstractAgent:
    if config.agent_type == "default":
        agents = [
            DefaultAgent(config, obs_shape, act_size)
            for obs_shape, act_size in zip(observation_space, action_space)
        ]

    elif config.agent_type == "attention":
        agents = [
            AttentionAgent(config, obs_shape, act_size)
            for obs_shape, act_size in zip(observation_space, action_space)
        ]
    
    elif config.agent_type == "RecurrentAgent":
        agents = [
            RecurrentAgent(config, obs_shape, act_size)
            for obs_shape, act_size in zip(observation_space, action_space)
        ]

    elif config.agent_type == "recurrent_attention":
        agents = [
            RecurrentAttentionAgent(config, obs_shape, act_size)
            for obs_shape, act_size in zip(observation_space, action_space)
        ]

    else:
        logger.warn(f"Unexpected agent is given. config.agent: {config.agent}")

        raise ValueError()

    return agents
