"""Source code for default test multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.environments.default_environment import DefaultEnvironment
from core.worlds.entity import Agent


class TestEnvironment(DefaultEnvironment):
    def reset(self):
        obs_n = super().reset()
        self.objects_completed_successfully = 0
        return obs_n

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0

        if hasattr(self.world.map, 'specified_object_types'):
            agent_tasks = self.world.map.specified_object_types[agent_id]
        else:
            agent_tasks = self.agent_tasks[agent_id]

        if agent_tasks == "-1":
            return reward

        for object_type in range(self.type_objects):
            if (
                self.world.map.objects_matrix[object_type, a_pos_x, a_pos_y]
                == 1
            ):
                if (
                    self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y] == 1 and str(object_type) in agent_tasks
                ):
                    reward = 1.0
                    self.objects_completed_successfully += 1
                self.world.map.objects_matrix[object_type, a_pos_x, a_pos_y] = 0
                self.objects_completed += 1
                self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1
                if self.config.keep_objects_num:
                    self.generate_objects(1, object_type)

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
