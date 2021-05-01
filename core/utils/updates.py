# -*- coding: utf-8 -*-

"""Source code for update method.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
