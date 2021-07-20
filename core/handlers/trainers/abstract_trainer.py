import random
from abc import ABC

import torch
import wandb
from core.utils.buffer import Experience, ReplayBuffer
from omegaconf import DictConfig
from core.utils.dataset import RLDataset
from torch.utils.data import DataLoader

from ..abstract_loop_handler import AbstractLoopHandler


class AbstractTrainer(AbstractLoopHandler, ABC):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config=config, environment=environment)
        self.buffer = ReplayBuffer(
            config.capacity,
            state_conv=config.model.name in ["conv_mlp", "mat", "mat_baseline"],
        )
        self.epsilon = config.epsilon_initial

        wandb.init(
            project="MAEXP",
            entity="yoshi-0921",
            name=config.name,
            config=dict(config),
            tags=[
                config.world + "_world",
                config.environment + "_environment",
                config.agent_type + "_agent",
                config.brain + "_brain",
                config.phase + "_pahse",
                config.trainer + "_trainer",
                config.model.name + "_model",
                config.map.name + "_map",
                config.view_method + "_method",
            ],
        )

    def loss_and_update(self, batch):
        loss_list = []
        states, actions, rewards, dones, next_states = batch
        for agent_id, agent in enumerate(self.agents):
            loss = agent.learn(
                states[agent_id],
                actions[agent_id],
                rewards[agent_id],
                dones[agent_id],
                next_states[agent_id],
            )
            loss_list.append(loss.detach().cpu())

        return torch.stack(loss_list)

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        for agent_id in self.order:
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action = self.agents[agent_id].get_action(states[agent_id], epsilon)
            actions[agent_id] = action

        rewards, dones, new_states = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return states, rewards

    def setup(self):
        # set dataloader
        dataset = RLDataset(self.buffer, self.config.batch_size)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
        )

        # populate buffer
        self.populate(self.config.populate_steps)
        self.reset()

        # log brain networks of agents
        self.log_models()
