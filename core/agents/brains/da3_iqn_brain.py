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
        if config.observation_area_mask == "merged":
            self.local_patched_size_x = (
                config.visible_range // 1 # config.model.local_patch_size
            )
            self.local_patched_size_y = (
                config.visible_range // 1 # config.model.local_patch_size
            )
            self.relative_patched_size_x = (
                config.map.SIZE_X // 5 # config.model.relative_patch_size
            )
            self.relative_patched_size_y = (
                config.map.SIZE_Y // 5 # config.model.relative_patch_size
            )

        else:
            setattr(self, f"{config.observation_area_mask}_patched_size_x", obs_shape[1] // config.model.patch_size)
            setattr(self, f"{config.observation_area_mask}_patched_size_y", obs_shape[2] // config.model.patch_size)

    @torch.no_grad()
    def get_action(self, state):
        for state_key, state_value in state.items():
            if isinstance(state_value, torch.Tensor):
                state[state_key] = state_value.unsqueeze(0).float().to(self.device)

        q_values, attns = self.network.forward_attn(state)

        action = q_values.argmax(dim=1)
        action = int(action.item())

        return action, attns
