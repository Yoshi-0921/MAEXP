"""Source code for dra3-iqn brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import torch
from torch import nn

from .da3_iqn_brain import DA3_IQNBrain
from .iqn_brain import calculate_quantile_huber_loss


class DRA3_IQNBrain(DA3_IQNBrain):
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
        hidden_vector = None
        target_sa_quantiles_seq, current_sa_quantiles_seq = [], []
        for states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind in zip(states_seq, actions_seq, rewards_seq, dones_seq, next_states_seq):
            for states_key, states_value in states_ind.items():
                if isinstance(states_value, torch.Tensor):
                    states_ind[states_key] = states_value.float().to(self.device)
            actions_ind = actions_ind.to(self.device)
            rewards_ind = rewards_ind.float().to(self.device)
            dones_ind = dones_ind.to(self.device)
            for next_states_key, next_states_value in next_states_ind.items():
                if isinstance(next_states_value, torch.Tensor):
                    next_states_ind[next_states_key] = next_states_value.float().to(
                        self.device
                    )
            if isinstance(hidden_vector, torch.Tensor):
                hidden_vector = hidden_vector.to(self.device)

            batch_size = dones_ind.shape[0]
            taus = torch.rand(batch_size, self.num_quantiles, device=self.device)
            current_s_quantiles = self.network.get_quantiles(state=states_ind, taus=taus)
            action_index = (
                actions_ind.unsqueeze(1)
                .unsqueeze(2)
                .expand(batch_size, self.num_quantiles, 1)
            )
            current_sa_quantiles = current_s_quantiles.gather(dim=2, index=action_index)
            current_sa_quantiles_seq.append(current_sa_quantiles)
            with torch.no_grad():
                # 最も価値の高いactionを抽出
                q_values, hidden_vector = self.network(next_states_ind, hidden_vector)
                best_actions = q_values.argmax(dim=1)

                tau_dashes = torch.rand(
                    batch_size, self.num_quantiles_dash, device=self.device
                )
                next_s_quantiles = self.target_network.get_quantiles(
                    state=next_states_ind, taus=tau_dashes
                )
                # terminal state以外はDDQNで計算したもので上書き
                best_action_index = (
                    best_actions.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(batch_size, self.num_quantiles_dash, 1)
                )
                next_sa_quantiles = next_s_quantiles.gather(
                    dim=2, index=best_action_index
                ).transpose(1, 2)

                # Calculate target quantile values.
                target_sa_quantiles = (
                    rewards_ind.unsqueeze(1).unsqueeze(2) + self.gamma * next_sa_quantiles
                )
                target_sa_quantiles_seq.append(target_sa_quantiles)

        self.optimizer.zero_grad()
        loss = 0
        for sequence_id, (target_sa_quantiles, current_sa_quantiles) in enumerate(zip(target_sa_quantiles_seq, current_sa_quantiles_seq)):
            td_errors = target_sa_quantiles - current_sa_quantiles
            loss += calculate_quantile_huber_loss(td_errors, taus) * int(len(target_sa_quantiles_seq) // 2 <= sequence_id)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        self.optimizer.step()

        return {"total_loss": loss.detach().cpu()}
