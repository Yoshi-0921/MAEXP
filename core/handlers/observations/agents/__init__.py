# -*- coding: utf-8 -*-

"""Builds agent observation handler used in multi-agent reinforcement learning.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from core.utils.logging import initialize_logging
from core.worlds import AbstractWorld

from .default_agent_observation import DefaultAgentObservationHandler
from .self_agent_observation import SelfAgentObservationHandler
from .simple_agent_observation import SimpleAgentObservationHandler
from .transition_agent_observation import TransitionAgentObservationHandler
from .individual_agent_observation import IndividualAgentObservationHandler

logger = initialize_logging(__name__)


def generate_observation_agent(config, world: AbstractWorld):
    if config.agent_view_method == "default":
        observation_agent = DefaultAgentObservationHandler(config=config, world=world)

    elif config.agent_view_method == "self":
        observation_agent = SelfAgentObservationHandler(config=config, world=world)

    elif config.agent_view_method == "simple":
        observation_agent = SimpleAgentObservationHandler(config=config, world=world)

    elif config.agent_view_method == "transition":
        observation_agent = TransitionAgentObservationHandler(config=config, world=world)

    elif config.agent_view_method == "individual":
        observation_agent = IndividualAgentObservationHandler(config=config, world=world)

    else:
        logger.warn(
            f"Unexpected agent_view_method is given. config.agent_view_method: {config.agent_view_method}"
        )

        raise ValueError()

    return observation_agent
