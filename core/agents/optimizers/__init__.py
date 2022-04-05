# -*- coding: utf-8 -*-

"""Builds optimizer used to update brain network of agents.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""
from typing import Iterator

from core.utils.logging import initialize_logging
from omegaconf import DictConfig
from torch import optim
from torch.nn.parameter import Parameter

logger = initialize_logging(__name__)


def generate_optimizer(config: DictConfig, parameters: Iterator[Parameter]):
    if config.opt.name == "adam":
        optimizer = optim.Adam(
            params=parameters,
            lr=config.opt.learning_rate,
            betas=config.opt.betas,
            eps=config.opt.eps,
        )

    elif config.opt.name == "rmsprop":
        optimizer = optim.RMSprop(
            params=parameters,
            lr=config.opt.learning_rate,
            alpha=config.opt.alpha,
            eps=config.opt.eps,
        )

    else:
        logger.warn(
            f"Unexpected optimizer name is given. config.opt.name: {config.opt.name}"
        )

        raise ValueError()

    return optimizer
