# -*- coding: utf-8 -*-

"""Source code for abstract brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from abc import ABC, abstractmethod
from typing import List

import torch
from core.agents.criterions import generate_criterion
from core.agents.models import generate_network
from core.agents.optimizers import generate_optimizer
from core.utils.logging import initialize_logging
from omegaconf import DictConfig

logger = initialize_logging(__name__)


class AbstractBrain(ABC):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        self.config = config
        self.network = generate_network(config=config, obs_shape=obs_shape, act_size=act_size)
        self.target_network = generate_network(config=config, obs_shape=obs_shape, act_size=act_size)
        self.criterion = generate_criterion(config)
        self.optimizer = generate_optimizer(config, self.network)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    @torch.no_grad()
    def get_action(self, state):
        raise NotImplementedError()

    @abstractmethod
    def learn(self, states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind):
        raise NotImplementedError()