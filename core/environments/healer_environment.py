"""Source code for healer-tasks multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.worlds.entity import Agent

from .default_environment import DefaultEnvironment


class HealerEnvironment(DefaultEnvironment):
    def reset(self):
        obs_n = super().reset()
        for agent_id, (agent, order) in enumerate(
            zip(self.agents, self.init_xys_order)
        ):
            agent.status["remains"] = 10
        return obs_n

    def generate_objects(self, num_objects: int = None, object_type: int = None):
        num_objects = num_objects or self.config.num_objects
        self._generate_objects(num_objects)

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0
        if self.agent_tasks[agent_id] == "-1":
            return reward

        if self.agent_tasks[agent_id] == "o":
            #視野内に他のエージェントがいれば正の報酬
            pass

        for object_type in self.agent_tasks[agent_id]:
            if (
                self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y]
                == self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y]
                == 1
            ):
                if agent.status["remains"] > 0:
                    agent.status["remains"] -= 1
                    reward = 1.0
                    self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y] = 0
                    self.objects_completed += 1
                    self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1
                    if self.config.keep_objects_num:
                        self.generate_objects(1, int(object_type))
                else:
                    reward = -0.1

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
