# -*- coding: utf-8 -*-

"""Source code for dqn brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from .abstract_brain import AbstractBrain
import torch
from torch import nn


class DQNBrain(AbstractBrain):
    @torch.no_grad()
    def get_action(self, state):
        state = state.unsqueeze(0).to(self.device)

        q_values = self.network(state)
        q_value, action = torch.max(q_values, dim=1)

        return action

    def learn(self, states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind):
        self.network.eval()
        self.target_network.eval()
        q_values = self.network(states_ind)
        state_action_values = q_values.gather(1, actions_ind.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            out = self.target_network(next_states_ind)
            next_state_values = out.max(1)[0]
            next_state_values[dones_ind] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = rewards_ind + self.gamma * (1 - dones_ind) * next_state_values

        self.network.train()
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        self.optimizer.step()

        return loss