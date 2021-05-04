# -*- coding: utf-8 -*-

"""Set seed for multi-agent experiments.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 921):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
