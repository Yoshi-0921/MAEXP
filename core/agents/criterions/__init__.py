# -*- coding: utf-8 -*-

"""Builds criterion to evaluate estimated action value of learning agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from omegaconf import DictConfig
from torch import nn


def generate_criterion(config: DictConfig):

    return nn.MSELoss()
