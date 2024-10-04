"""Source code for individual-transition observation handler for agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch
from omegaconf import DictConfig

from core.worlds import AbstractWorld

from ..abstract_observation_handler import AbstractObservationHandler
from .individual_agent_observation import IndividualAgentObservationHandler
from .transition_agent_observation import TransitionAgentObservationHandler


class IndividualTransitionAgentObservationHandler(AbstractObservationHandler):
    def __init__(self, config: DictConfig, world: AbstractWorld):
        super().__init__(config=config, world=world)
        self.individual_agent_observation = IndividualAgentObservationHandler(config=config, world=world)
        self.transition_agent_observation = TransitionAgentObservationHandler(config=config, world=world)

    def get_channel(self):
        return self.individual_agent_observation.get_channel() + self.transition_agent_observation.get_channel()

    def fill(self, agents, agent, agent_id, area_mask, coordinates):
        individual_agent_obs = self.individual_agent_observation.fill(agents=agents, agent=agent, agent_id=agent_id, area_mask=area_mask, coordinates=coordinates)
        transition_agent_obs = self.transition_agent_observation.fill(agents=agents, agent=agent, agent_id=agent_id, area_mask=area_mask, coordinates=coordinates)
        return torch.cat([individual_agent_obs, transition_agent_obs])

    def render(self, obs, image, channel):
        image, channel = self.individual_agent_observation.render(obs=obs, image=image, channel=channel)
        return self.transition_agent_observation.render(obs=obs, image=image, channel=channel)

    def step(self, agents, coordinate_handler):
        self.individual_agent_observation.step(agents=agents, coordinate_handler=coordinate_handler)
        self.transition_agent_observation.step(agents=agents, coordinate_handler=coordinate_handler)

    def reset(self, agents, coordinate_handler):
        self.individual_agent_observation.reset(agents=agents, coordinate_handler=coordinate_handler)
        self.transition_agent_observation.reset(agents=agents, coordinate_handler=coordinate_handler)
