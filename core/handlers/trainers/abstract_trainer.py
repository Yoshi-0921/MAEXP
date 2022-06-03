import random
from abc import ABC
from copy import deepcopy

import torch
import wandb
from core.utils.buffer import Experience, generate_buffer
from core.utils.dataset import RLDataset
from omegaconf import DictConfig
from typing import Dict, List
from torch.utils.data import DataLoader

from ..abstract_loop_handler import AbstractLoopHandler


class AbstractTrainer(AbstractLoopHandler, ABC):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config=config, environment=environment)
        self.buffer = generate_buffer(config=config)
        self.epsilon = config.epsilon_initial
        self.synchronize_frequency = config.synchronize_frequency

        wandb.init(
            project=config.project_name,
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
                config.observation_area_mask + "_area_mask",
                config.agent_view_method + "_agent_view_method",
                config.object_view_method + "_object_view_method",
            ],
        )

    def loss_and_update(self, batch) -> List[Dict[str, torch.Tensor]]:
        loss_list = []
        states, actions, rewards, dones, next_states = batch
        for agent_id, agent in enumerate(self.agents):
            if self.config.agent_tasks[int(agent_id)] == "-1":
                loss_list.append({"total_loss": torch.zeros(size=(1,))[0]})
                continue
            loss = agent.learn(
                states[agent_id],
                actions[agent_id],
                rewards[agent_id],
                dones[agent_id],
                next_states[agent_id],
            )
            loss_list.append(loss)

        return loss_list

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states
        for agent_id in self.order:
            if self.config.agent_tasks[int(agent_id)] == "-1":
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue
            action = self.agents[agent_id].get_action(
                deepcopy(states[agent_id]), epsilon
            )
            actions[agent_id] = action

        rewards, dones, new_states = self.env.step(actions, self.order)

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
