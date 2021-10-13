# -*- coding: utf-8 -*-

"""Source code for multi-agent types environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.worlds.entity import Agent

from .default_environment import DefaultEnvironment


class TypesEnvironment(DefaultEnvironment):
    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        if agent_id in [4, 5]:
            return 0.0
        else:
            return super().reward_ind(agents, agent, agent_id)
