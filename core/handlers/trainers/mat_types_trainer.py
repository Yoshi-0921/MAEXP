import random

import torch
from core.utils.buffer import Experience

from .mat_trainer import MATTrainer


class MATTypesTrainer(MATTrainer):
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states.clone()
        for agent_id in self.order:
            if agent_id in [0, 2]:
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue

            action, attns = self.agents[agent_id].get_action_attns(
                states[agent_id], epsilon
            )
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        rewards, dones, new_states = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return states, rewards, attention_maps
