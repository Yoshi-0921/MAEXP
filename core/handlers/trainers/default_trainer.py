import random

import numpy as np
import torch
import wandb
from core.agents.random_agent import RandomAgent
from core.utils.buffer import Experience
from core.utils.dataset import RLDataset
from core.utils.updates import hard_update
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .abstract_trainer import AbstractTrainer


class DefaultTrainer(AbstractTrainer):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config=config, environment=environment)
        self.visible_range = self.config.visible_range
        wandb.init(
            project="MAEXP", entity="yoshi-0921", name="default", config=dict(config)
        )

    def generate_agents(self):
        obs_size = self.env.observation_space
        act_size = self.env.action_space
        agents = [
            RandomAgent(obs_size[agent_id], act_size[agent_id], self.config)
            for agent_id in range(self.env.num_agents)
        ]

        return agents

    def loss_and_update(self, batch):
        loss = list()

        return loss

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        for agent_id in self.order:
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action = self.agents[agent_id].get_action(states[agent_id], epsilon)
            actions[agent_id] = action

        new_states, rewards, dones = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return states, rewards

    def setup(self):
        # set dataloader
        dataset = RLDataset(self.buffer, self.config.batch_size)
        self.dataloader = DataLoader(
            dataset=dataset, batch_size=self.config.batch_size, pin_memory=True
        )

        # put models on GPU and change to training mode
        for agent in self.agents:
            agent.dqn.to(self.device)
            agent.dqn_target.to(self.device)
            agent.dqn.train()
            agent.dqn_target.eval()

        # populate buffer
        self.populate(self.config.populate_steps)
        self.reset()

    def training_epoch_start(self, epoch: int):
        if epoch % (self.config.max_epochs // 10) == 0:
            for agent_id, agent in enumerate(self.agents):
                model_path = f"epoch{epoch}_agent{agent_id}.pth"
                torch.save(agent.dqn.to("cpu").state_dict(), model_path)
                agent.dqn.to(self.device)

    def training_step(self, step: int, epoch: int):
        # train based on experiments
        for batch in self.dataloader:
            loss_list = self.loss_and_update(batch)
            self.total_loss_sum += torch.sum(loss_list)

        # execute in environment
        states, rewards = self.play_step(self.epsilon)
        self.episode_reward += np.sum(rewards)

        if epoch % 10 == 0 and step % (self.config.max_episode_length // 5) == 0:
            # log attention_maps of agent0
            for agent_id in range(len(self.agents)):
                state = F.interpolate(
                    states, size=(self.visible_range * 20, self.visible_range * 20)
                )[agent_id]
                image = torch.zeros(
                    (3, self.visible_range * 20, self.visible_range * 20),
                    dtype=torch.float,
                )

                # agentの情報を追加(Blue)
                image[2, ...] += state[0]
                # landmarkの情報を追加(Yellow)
                image[0, ...] += state[1]
                image[1, ...] += state[1]
                # invisible areaの情報を追加(White)
                image[:, ...] -= state[2]

                wandb.log(
                    {
                        f"observation_{str(agent_id)}": [
                            wandb.Image(
                                data_or_path=image[:, :, [2, 1, 0]],
                                caption="local observation",
                            )
                        ]
                    },
                    step=self.global_step,
                )

        wandb.log(
            {
                'training/epsilon': torch.tensor(self.epsilon),
                "training/reward": torch.tensor(rewards).mean(),
                "training/total_loss": self.total_loss_sum
            },
            step=self.global_step
        )

    def training_epoch_end(self):
        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.config.epsilon_end, self.epsilon)

        # update target network
        for agent in self.agents:
            hard_update(agent.dqn_target, agent.dqn)

        self.reset()

    def endup(self):
        self.writer.close()

        for agent_id, agent in enumerate(self.agents):
            model_path = f"agent_{agent_id}.pth"
            torch.save(agent.dqn.to("cpu").state_dict(), model_path)
