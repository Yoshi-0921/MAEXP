"""Source code for categorical dqn brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from .abstract_brain import AbstractBrain
from core.agents.models.customs.categorical_dqn import ApplySoftmax


class CategoricalDQNBrain(AbstractBrain):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(config=config, obs_shape=obs_shape, act_size=act_size)
        self.gamma = config.gamma
        self.num_atoms = config.model.num_atoms
        self.V_min = config.model.V_min
        self.V_max = config.model.V_max
        self.support = torch.linspace(self.V_min, self.V_max, self.num_atoms).to(
            device=self.device
        )  # Support (range) of z
        self.delta_z = (self.V_max - self.V_min) / (self.num_atoms - 1)

    @torch.no_grad()
    def get_action(self, state):
        for state_key, state_value in state.items():
            state[state_key] = state_value.unsqueeze(0).float().to(self.device)

        model_output = self.network(state, ApplySoftmax.NORMAL)
        model_output = torch.sum(model_output * self.support, dim=2)

        _, action = torch.max(model_output, dim=1)
        action = int(action.item())

        return action

    def learn(self, states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind):
        for states_key, states_value in states_ind.items():
            states_ind[states_key] = states_value.float().to(self.device)
        actions_ind = actions_ind.to(self.device)
        rewards_ind = rewards_ind.float().to(self.device)
        dones_ind = dones_ind.to(self.device)
        for next_states_key, next_states_value in next_states_ind.items():
            next_states_ind[next_states_key] = next_states_value.float().to(self.device)

        batch_size = dones_ind.shape[0]
        log_p = self.network(states_ind, ApplySoftmax.LOG)
        log_p_a = log_p[range(batch_size), actions_ind.squeeze()]
        with torch.no_grad():
            # 最も価値の高いactionを抽出
            model_output = self.network(next_states_ind, ApplySoftmax.NORMAL)
            best_actions = torch.sum(model_output * self.support, dim=2).argmax(dim=1)

            p_next = self.target_network(next_states_ind, ApplySoftmax.NORMAL)

            # (1) terminal state用に確率分布としてすべてのatomに同じ値を与えておく
            p_next_best = torch.zeros(0).to(self.device, dtype=torch.float32).new_full((batch_size, self.num_atoms), 1.0 / self.num_atoms)
            # terminal state以外はDDQNで計算したもので上書き
            p_next_best = p_next[range(batch_size), best_actions]

            # 報酬を分布に直す
            Tz = (rewards_ind.unsqueeze(1) + self.gamma * self.support.unsqueeze(0)).clamp(self.V_min, self.V_max)
            b = (Tz - self.V_min) / self.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()

            # (3) bの値がちょうど整数値だった場合にmの要素値が0となってしまうことを回避
            lower[(lower == upper) * (0 < lower)] -= 1
            # ↑の処理によってlの値は既に変更済みなため、↓の処理が同時に行われてしまうことはない
            upper[(lower == upper) * (upper < self.num_atoms - 1)] += 1

            m = torch.zeros(batch_size, self.num_atoms).to(self.device, dtype=torch.float32)
            # (4) ミニバッチの各要素毎に和を持っておくため、offsetを計算した上でmを一次元のリストにして扱う
            offset = torch.linspace(0, ((batch_size - 1) * self.num_atoms), batch_size).unsqueeze(1).expand(batch_size, self.num_atoms).to(lower)
            m.view(-1).index_add_(0, (lower + offset).view(-1), (p_next_best * (upper.float() - b)).view(-1))
            m.view(-1).index_add_(0, (upper + offset).view(-1), (p_next_best * (b - lower.float())).view(-1))

        self.optimizer.zero_grad()
        loss = -torch.sum(m * log_p_a, dim=1).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()

        return loss.detach()
