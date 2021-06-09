# -*- coding: utf-8 -*-

"""Source code for dqn brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from .abstract_brain import AbstractBrain


class DQNBrain(AbstractBrain):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(config=config, obs_shape=obs_shape, act_size=act_size)
        self.gamma = config.gamma

    @torch.no_grad()
    def get_action(self, state):
        state = state.unsqueeze(0).float().to(self.device)

        q_values = self.network(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())

        return action

    def learn(self, states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind):
        states_ind = states_ind.float().to(self.device)
        actions_ind = actions_ind.to(self.device)
        rewards_ind = rewards_ind.to(self.device)
        dones_ind = dones_ind.to(self.device)
        next_states_ind = next_states_ind.float().to(self.device)

        self.network.eval()
        self.target_network.eval()
        q_values = self.network(states_ind)
        state_action_values = q_values.gather(1, actions_ind.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            out = self.target_network(next_states_ind)
            next_state_values = out.max(1)[0]
            next_state_values[dones_ind] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = (
            rewards_ind + self.gamma * (1 - dones_ind) * next_state_values
        )

        self.network.train()
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        self.optimizer.step()

        return loss.detach()
