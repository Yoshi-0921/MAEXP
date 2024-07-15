"""Source code for fully parameterized function (FQF) brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig
from torch import nn

from core.agents.optimizers import generate_optimizer

from .abstract_brain import AbstractBrain
from .iqn_brain import calculate_quantile_huber_loss


class FQFBrain(AbstractBrain):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(config=config, obs_shape=obs_shape, act_size=act_size)
        self.gamma = config.gamma
        self.num_quantiles = config.model.num_quantiles
        self.entropy_coef = config.model.entropy_coef
        self.optimizer = generate_optimizer(
            config,
            list(self.network.conv.parameters())
            + list(self.network.state_embedder.parameters())
            + list(self.network.cosine_net.parameters())
            + list(self.network.fc_V.parameters())
            + list(self.network.fc_A.parameters()),
        )
        config.opt.learning_rate *= 1e-6
        self.fraction_optimizer = generate_optimizer(
            config, self.network.fraction_net.parameters()
        )

    def learn(self, states_ind, actions_ind, rewards_ind, dones_ind, next_states_ind):
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

        batch_size = dones_ind.shape[0]
        state_embeddings = self.network.get_state_embeddings(state=states_ind)
        taus, tau_hats, entropies = self.network.get_taus(
            state_embeddings=state_embeddings.detach()
        )
        current_s_quantile_hats = self.network.get_quantiles(
            state_embeddings=state_embeddings, taus=tau_hats
        )
        action_index = (
            actions_ind.unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, self.num_quantiles, 1)
        )
        current_sa_quantile_hats = current_s_quantile_hats.gather(
            dim=2, index=action_index
        )

        fraction_loss = self.learn_fractions(
            state_embeddings=state_embeddings.detach(),
            sa_quantile_hats=current_sa_quantile_hats.detach(),
            taus=taus,
            actions=actions_ind,
        )
        entropy_loss = entropies.mean()
        fraction_total_loss = fraction_loss - self.entropy_coef * entropy_loss

        # Calculates quantile loss
        with torch.no_grad():
            # 最も価値の高いactionを抽出
            q_values = self.network(state=next_states_ind)
            best_actions = q_values.argmax(dim=1)

            next_s_quantile_hats = self.target_network.get_quantiles(
                state=next_states_ind, taus=tau_hats
            )
            # terminal state以外はDDQNで計算したもので上書き
            best_action_index = (
                best_actions.unsqueeze(1)
                .unsqueeze(2)
                .expand(batch_size, self.num_quantiles, 1)
            )
            next_sa_quantile_hats = next_s_quantile_hats.gather(
                dim=2, index=best_action_index
            ).transpose(1, 2)

            # Calculate target quantile values.
            target_sa_quantile_hats = (
                rewards_ind.unsqueeze(1).unsqueeze(2)
                + self.gamma * next_sa_quantile_hats
            )

        self.fraction_optimizer.zero_grad()
        fraction_total_loss.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(
        #     self.network.fraction_net.parameters(),
        #     0.1,
        # )
        self.fraction_optimizer.step()

        self.optimizer.zero_grad()
        td_errors = target_sa_quantile_hats - current_sa_quantile_hats
        loss = calculate_quantile_huber_loss(td_errors, tau_hats)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.network.conv.parameters())
            + list(self.network.state_embedder.parameters())
            + list(self.network.cosine_net.parameters())
            + list(self.network.fc_V.parameters())
            + list(self.network.fc_A.parameters()),
            0.1,
        )
        self.optimizer.step()

        return {
            "total_loss": fraction_loss.detach().cpu() + loss.detach().cpu(),
            "fraction_loss": fraction_loss.detach().cpu(),
            "entropy_loss": entropy_loss.detach().cpu(),
            "quantile_loss": loss.detach().cpu()
        }

    def learn_fractions(self, state_embeddings, sa_quantile_hats, taus, actions):
        assert not state_embeddings.requires_grad
        assert not sa_quantile_hats.requires_grad

        batch_size = state_embeddings.shape[0]

        with torch.no_grad():
            quantiles = self.network.get_quantiles(
                state_embeddings=state_embeddings, taus=taus[:, 1:-1]
            )
            action_index = (
                actions.unsqueeze(1)
                .unsqueeze(2)
                .expand(batch_size, self.num_quantiles - 1, 1)
            )
            sa_quantiles = quantiles.gather(dim=2, index=action_index)

            assert sa_quantiles.shape == (batch_size, self.num_quantiles - 1, 1)

        # NOTE: Proposition 1 in the paper requires F^{-1} is non-decreasing.
        # I relax this requirements and calculate gradients of taus even when
        # F^{-1} is not non-decreasing.

        values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
        signs_1 = sa_quantiles > torch.cat(
            [sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1
        )
        assert values_1.shape == signs_1.shape

        values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
        signs_2 = sa_quantiles < torch.cat(
            [sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1
        )
        assert values_2.shape == signs_2.shape

        gradient_of_taus = (
            torch.where(signs_1, values_1, -values_1)
            + torch.where(signs_2, values_2, -values_2)
        ).view(batch_size, self.num_quantiles - 1)
        assert not gradient_of_taus.requires_grad
        assert gradient_of_taus.shape == taus[:, 1:-1].shape

        fraction_loss = (gradient_of_taus * taus[:, 1:-1]).sum(dim=1).mean()

        return fraction_loss
