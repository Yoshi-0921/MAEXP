# -*- coding: utf-8 -*-

"""Source code to set up activation function list.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from torch.nn import functional as F

from core.utils.logging import initialize_logging

logger = initialize_logging(__name__)


def add_activation_functions(activation_functions):
    af_list = []
    for activation_function in activation_functions:
        if activation_function == "relu":
            af_list.append(F.relu)

        elif activation_function == "sigmoid":
            af_list.append(F.sigmoid)

        elif activation_function == "tanh":
            af_list.append(F.tanh)

        elif activation_function == "max_pool2d":
            af_list.append(F.max_pool2d)

        else:
            logger.warn(
                f"Unexpected activation function is given. activation_function: {activation_function}"
            )

            raise ValueError()

    return af_list
