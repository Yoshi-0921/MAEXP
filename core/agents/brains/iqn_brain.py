"""Source code for implicit quantile network (IQN) brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from .abstract_brain import AbstractBrain


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa),
    )


def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = (
        torch.abs(taus[..., None] - (td_errors.detach() < 0).float())
        * element_wise_huber_loss
        / kappa
    )
    assert element_wise_quantile_huber_loss.shape == (batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(
        dim=1, keepdim=True
    )
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


class IQNBrain(AbstractBrain):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(config=config, obs_shape=obs_shape, act_size=act_size)
        self.gamma = config.gamma
        self.num_quantiles = config.model.num_quantiles
        self.num_quantiles_dash = config.model.num_quantiles_dash

    @torch.no_grad()
    def get_action(self, state):
        for state_key, state_value in state.items():
            state[state_key] = state_value.unsqueeze(0).float().to(self.device)

        taus = torch.rand(1, self.num_quantiles, device=self.device)
        quantiles = self.network(state, taus)
        q_values = quantiles.mean(dim=2)

        _, action = torch.max(q_values, dim=1)
        action = int(action.item())

        return action

    def learn(self, states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind):
        for states_key, states_value in states_ind.items():
            if isinstance(states_value, torch.Tensor):
                states_ind[states_key] = states_value.float().to(self.device)
        actions_ind = actions_ind.to(self.device)
        rewards_ind = rewards_ind.float().to(self.device)
        dones_ind = dones_ind.to(self.device)
        for next_states_key, next_states_value in next_states_ind.items():
            if isinstance(next_states_value, torch.Tensor):
                next_states_ind[next_states_key] = next_states_value.float().to(self.device)

        batch_size = dones_ind.shape[0]
        taus = torch.rand(batch_size, self.num_quantiles, device=self.device)
        current_s_quantiles = self.network(states_ind, taus)
        current_sa_quantiles = current_s_quantiles[
            range(batch_size), actions_ind.squeeze()
        ].unsqueeze(1)
        with torch.no_grad():
            # 最も価値の高いactionを抽出
            quantiles = self.network(next_states_ind, taus)  # ここのtausは再度生成した方が良いかも？ ← そうなら内部に組み込める
            best_actions = quantiles.mean(dim=2).argmax(dim=1)

            tau_dashes = torch.rand(
                batch_size, self.num_quantiles_dash, device=self.device
            )
            p_next = self.target_network(next_states_ind, tau_dashes)

            # terminal state以外はDDQNで計算したもので上書き
            next_sa_quantiles = p_next[range(batch_size), best_actions].unsqueeze(1).transpose(1, 2)

            # Calculate target quantile values.
            target_sa_quantiles = (
                rewards_ind.unsqueeze(1).unsqueeze(2) + self.gamma * next_sa_quantiles
            )

        self.optimizer.zero_grad()
        td_errors = target_sa_quantiles - current_sa_quantiles
        loss = calculate_quantile_huber_loss(td_errors, taus)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()

        return loss.detach()
