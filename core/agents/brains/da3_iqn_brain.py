"""Source code for da3-iqn brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

import torch
from omegaconf import DictConfig

from .iqn_brain import IQNBrain


class DA3_IQNBrain(IQNBrain):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(config=config, obs_shape=obs_shape, act_size=act_size)
        setattr(self, f"{config.observation_area_mask}_patched_size_x", obs_shape[1] // config.model.patch_size)
        setattr(self, f"{config.observation_area_mask}_patched_size_y", obs_shape[2] // config.model.patch_size)

    @torch.no_grad()
    def get_action(self, state):
        for state_key, state_value in state.items():
            state[state_key] = state_value.unsqueeze(0).float().to(self.device)

        taus = torch.rand(1, self.num_quantiles, device=self.device)
        quantiles, attns = self.network.forward_attn(state, taus)
        q_values = quantiles.mean(dim=2)

        _, action = torch.max(q_values, dim=1)
        action = int(action.item())

        return action, attns
