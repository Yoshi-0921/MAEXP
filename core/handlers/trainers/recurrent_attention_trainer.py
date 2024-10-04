import random
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from omegaconf import DictConfig

from core.utils.buffer import Experience

from .attention_trainer import AttentionTrainer

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
sns.set()


class RecurrentAttentionTrainer(AttentionTrainer):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config=config, environment=environment)
        self.hidden_vectors = [torch.rand(1, 64) for _ in range(self.env.num_agents)]

    def loop_epoch_start(self, epoch: int):
        self.hidden_vectors = [torch.rand(1, 64) for _ in range(self.env.num_agents)]
        if epoch == (self.max_epochs // 2):
            self.save_state_dict(epoch=epoch)

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states
        for agent_id in self.order:
            if self.config.agent_tasks[int(agent_id)] == "-1":
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue

            action, attns, self.hidden_vectors[agent_id] = self.agents[agent_id].get_action_attns(
                deepcopy(states[agent_id]), epsilon, self.hidden_vectors[agent_id]
            )
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        rewards, dones, new_states = self.env.step(actions, self.order)

        exp = Experience(states, actions, rewards, dones, new_states)

        self.buffer.append(deepcopy(exp))

        del exp

        self.states = new_states

        return states, rewards, dones, attention_maps
