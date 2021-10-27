# -*- coding: utf-8 -*-

"""Source code for multi-agent world.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from abc import ABC
from typing import List

import numpy as np
from core.maps.abstract_map import AbstractMap
from .entity import Agent, Object
from omegaconf import DictConfig


class AbstractWorld(ABC):
    def __init__(self, config: DictConfig, world_map: AbstractMap):
        self.config = config
        self.map = world_map
        self.agents: List[Agent] = []
        self.objects: List[Object] = []

    @property
    def entities(self):
        return self.agents + self.objects

    def reset_all(self):
        self.reset_agents()
        self.reset_objects()
        self.reset_map()

    def reset_agents(self):
        self.agents = []

    def reset_objects(self):
        self.objects = []

    def reset_map(self):
        self.map.reset_all()

    def step(self, order):
        force = [None] * len(self.agents)
        force = self.apply_action_force(force)
        for agent_id in order:
            agent = self.agents[agent_id]
            force = self.apply_environment_force(agent_id, agent, force)
            agent.push(force[agent_id])
        self.integrate_state()

    def apply_action_force(self, force):
        for agent_id, agent in enumerate(self.agents):
            force[agent_id] = agent.action
        return force

    def apply_environment_force(self, agent_id, agent, force):
        # Check if there is a collision against other agents.
        next_pos = agent.xy + force[agent_id]
        for i, a in enumerate(self.agents):
            if agent_id == i:
                continue

            if all(next_pos == a.xy):
                force[agent_id] = np.zeros(2, dtype=np.int8)
                agent.collide_agents = True
                break

        # Check if there is a collision against wall.
        next_pos_x, next_pos_y = self.map.coord2ind(next_pos)
        if self.map.wall_matrix[next_pos_x, next_pos_y] == 1:
            force[agent_id] = np.zeros(2, dtype=np.int8)
            agent.collide_walls = True

        return force

    def integrate_state(self):
        self.map.reset_agents()
        for agent in self.agents:
            agent_x, agent_y = self.map.coord2ind(agent.xy)
            self.map.agents_matrix[agent_x, agent_y] = 1

    def render(self):
        render_list = []
        for y, ys in enumerate(self.map.wall_matrix.T):
            render_row = []
            for x, x_value in enumerate(ys):
                cx, cy = self.map.ind2coord((x, y))
                if x_value == 1:
                    render_row.append("#")
                elif cx == cy == 0:
                    render_row.append("+")
                elif cy == 0:
                    render_row.append("-")
                elif cx == 0:
                    render_row.append("|")
                else:
                    render_row.append(" ")
            render_list.append(render_row)

        for agent_id, agent in enumerate(self.agents):
            a_pos_x, a_pos_y = self.map.coord2ind(agent.xy)
            render_list[a_pos_y][a_pos_x] = str(agent_id)

        print(f"Num of Agents: {len(self.agents)}")
        print(f"SIZE_X: {self.map.SIZE_X}, SIZE_Y: {self.map.SIZE_Y}")
        for render_row in render_list:
            print(*render_row)
