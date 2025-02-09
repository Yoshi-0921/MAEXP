"""Source code for observer-tasks multi-agent environment.

Author: Yoshinari Motokawa <yoshinari.moto@fuji.waseda.jp>
"""

from typing import List

from core.worlds.entity import Agent

from .default_environment import DefaultEnvironment


class ObserverEnvironment(DefaultEnvironment):
    def generate_objects(self, num_objects: int = None, object_type: int = None):
        num_objects = num_objects or self.config.num_objects
        self._generate_objects(num_objects)

    def reward_ind(self, agents: List[Agent], agent: Agent, agent_id: int):
        agent_x, agent_y = agent.xy
        a_pos_x, a_pos_y = self.world.map.coord2ind(agent.xy)
        self.heatmap_agents[agent_id, a_pos_x, a_pos_y] += 1

        reward = 0.0
        if self.agent_tasks[agent_id] == "-1":
            return reward

        if self.agent_tasks[agent_id] == "o":
            for a_i, a in enumerate(agents[:-3]):
                a_x, a_y = a.xy
                a_p_x, a_p_y = self.world.map.coord2ind(a.xy)
                if (a_x - 3 <= agent_x <= a_x + 3) and (a_y - 3 <= agent_y <= a_y + 3):
                    for object_type in self.agent_tasks[a_i]:
                        if (
                            self.world.map.objects_matrix[int(object_type), a_p_x, a_p_y]
                            == self.world.map.destination_area_matrix[a_i][a_p_x, a_p_y]
                            == 1
                        ):
                            reward = 1.0

        else:
            is_within_observer_sight = False
            for observer in agents[-3:]:
                observer_x, observer_y = observer.xy
                if (observer_x - 3 <= agent_x <= observer_x + 3) and (observer_y - 3 <= agent_y <= observer_y + 3):
                    is_within_observer_sight = True

            for object_type in self.agent_tasks[agent_id]:
                if (
                    self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y]
                    == self.world.map.destination_area_matrix[agent_id][a_pos_x, a_pos_y]
                    == 1
                ):
                    reward = 0.1
                    if is_within_observer_sight:
                        reward = 1.0
                    self.world.map.objects_matrix[int(object_type), a_pos_x, a_pos_y] = 0
                    self.objects_completed += 1
                    self.heatmap_complete[agent_id, a_pos_x, a_pos_y] += 1
                    if self.config.keep_objects_num:
                        self.generate_objects(1, int(object_type))

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
