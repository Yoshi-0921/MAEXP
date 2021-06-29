# -*- coding: utf-8 -*-

"""Source code for mast brain class.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import List

import torch
from omegaconf import DictConfig
from .mat_brain import MATBrain


class MASTBrain(MATBrain):
    """"""


class MASTBaselineBrain(MASTBrain):
    def __init__(self, config: DictConfig, obs_shape: List[int], act_size: int):
        super().__init__(config=config, obs_shape=obs_shape, act_size=act_size)
        attn_size = self.patched_size_x * self.patched_size_y + 1
        self.attns = [
            torch.zeros(config.batch_size, config.model.num_heads, attn_size, attn_size).to(self.device)
        ]

    @torch.no_grad()
    def get_action(self, state):
        state = state.unsqueeze(0).to(self.device)

        q_values = self.network.forward(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())

        return action, self.attns
