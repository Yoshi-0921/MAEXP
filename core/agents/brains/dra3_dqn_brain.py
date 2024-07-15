"""Source code for dra3 dqn brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from .abstract_brain import AbstractBrain
from .da3_brain import DA3Brain


class DRA3_DQNBrain(DA3Brain):
    @torch.no_grad()
    def get_action(self, state, hidden_vector=None):
        for state_key, state_value in state.items():
            if isinstance(state_value, torch.Tensor):
                state[state_key] = state_value.unsqueeze(0).float().to(self.device)
        if isinstance(hidden_vector, torch.Tensor):
            hidden_vector = hidden_vector.to(self.device)

        q_values, attns, hidden_vector = self.network.forward_attn(state, hidden_vector=hidden_vector)

        action = q_values.argmax(dim=1)
        action = int(action.item())

        return action, attns, hidden_vector
    def learn(self, states_seq, actions_seq, rewards_seq, dones_seq, next_states_seq):
        hidden_vector, target_hidden_vector = None, None
        state_action_values_seq, expected_state_action_values_seq = [], []
        for states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind in zip(states_seq, actions_seq, rewards_seq, dones_seq, next_states_seq):
            for states_key, states_value in states_ind.items():
                if isinstance(states_value, torch.Tensor):
                    states_ind[states_key] = states_value.float().to(self.device)
            actions_ind = actions_ind.to(self.device)
            rewards_ind = rewards_ind.float().to(self.device)
            dones_ind = dones_ind.to(self.device)
            for next_states_key, next_states_value in next_states_ind.items():
                if isinstance(next_states_value, torch.Tensor):
                    next_states_ind[next_states_key] = next_states_value.float().to(self.device)
            if isinstance(hidden_vector, torch.Tensor):
                hidden_vector = hidden_vector.to(self.device)

            self.network.eval()
            self.target_network.eval()
            q_values, hidden_vector = self.network(states_ind, hidden_vector)
            state_action_values = q_values.gather(1, actions_ind.unsqueeze(-1)).squeeze(-1)
            state_action_values_seq.append(state_action_values)
            with torch.no_grad():
                out, _ = self.target_network(next_states_ind, hidden_vector.detach().clone())
                next_state_values = out.max(1)[0]
                next_state_values[dones_ind] = 0.0
                next_state_values = next_state_values.detach()
            expected_state_action_values = (
                rewards_ind + self.gamma * (1 - dones_ind.long()) * next_state_values
            )
            expected_state_action_values_seq.append(expected_state_action_values)

        self.network.train()
        self.optimizer.zero_grad()
        loss = 0
        for sequence_id, (expected_state_action_values, state_action_values) in enumerate(zip(expected_state_action_values_seq, state_action_values_seq)):
            loss += self.criterion(state_action_values, expected_state_action_values) * int(len(expected_state_action_values_seq) // 2 <= sequence_id)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        self.optimizer.step()

        return {"total_loss": loss.detach().cpu()}
