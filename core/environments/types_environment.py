# -*- coding: utf-8 -*-

"""Source code for multi-agent types environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.worlds.entity import Agent

from .default_environment import DefaultEnvironment


class TypesEnvironment(DefaultEnvironment):
    def generate_objects(self, num_objects: int = None, object_type: int = 0):
        if num_objects is None:
            for object_type in range(self.type_objects):
                self._generate_objects(
                    self.config.num_objects,
                    object_type=object_type,
                )

        else:
            self._generate_objects(1, object_type)

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0
        for obj_idx, obj in enumerate(self.world.objects):
            if all(agent.xy == obj.xy) and (
                (agent_id < 3 and obj.type == 0) or (3 <= agent_id and obj.type == 1)
            ):
                obj_pos_x, obj_pos_y = self.world.map.coord2ind(obj.xy)
                if (
                    self.world.map.destination_area_matrix[agent_id][
                        obj_pos_x, obj_pos_y
                    ]
                    == 1
                ):
                    reward = 1.0

                self.world.objects.pop(obj_idx)
                self.world.map.objects_matrix[obj.type, obj_pos_x, obj_pos_y] = 0
                self.objects_completed += 1
                self.heatmap_complete[agent_id, obj_pos_x, obj_pos_y] += 1
                self.generate_objects(1, obj.type)

        # negative reward for collision with other agents
        if agent.collide_agents:
            reward = -1.0
            agent.collide_agents = False
            self.heatmap_agents_collision[a_pos_x, a_pos_y] += 1
            self.agents_collided += 1

        # negative reward for collision against walls
        if agent.collide_walls:
            reward = -1.0
            agent.collide_walls = False
            self.heatmap_wall_collision[a_pos_x, a_pos_y] += 1
            self.walls_collided += 1

        return reward
