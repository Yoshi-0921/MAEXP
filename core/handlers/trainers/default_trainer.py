import random

import numpy as np
import torch
from core.agents.random_agent import RandomAgent
from core.utils.buffer import Experience
from core.utils.dataset import RLDataset
from core.utils.updates import hard_update
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .abstract_trainer import AbstractTrainer


class DefaultTrainer(AbstractTrainer):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config=config, environment=environment)
        self.visible_range = self.config.visible_range

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
        actions = [[] for i in range(self.env.num_agents)]
        attention_maps = [[] for i in range(self.env.num_agents)]
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

        return states, rewards, attention_maps

    def run(self):
        # set dataloader
        dataset = RLDataset(self.buffer, self.config.batch_size)
        dataloader = DataLoader(
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

        # training loop
        with tqdm(total=self.config.max_epochs) as pbar:
            for epoch in range(self.config.max_epochs):
                if epoch % (self.config.max_epochs // 10) == 0:
                    for agent_id, agent in enumerate(self.agents):
                        model_path = f"epoch{epoch}_agent{agent_id}.pth"
                        torch.save(agent.dqn.to("cpu").state_dict(), model_path)
                        agent.dqn.to(self.device)
                # training phase
                for step in range(self.config.max_episode_length):
                    total_loss_sum = 0.0

                    # train based on experiments
                    for batch in dataloader:
                        loss_list = self.loss_and_update(batch)
                        total_loss_sum += torch.sum(loss_list)

                    # execute in environment
                    states, rewards, attention_maps = self.play_step(self.epsilon)
                    self.episode_reward += np.sum(rewards)

                    if (
                        epoch % 10 == 0
                        and step % (self.config.max_episode_length // 5) == 0
                        and False
                    ):
                        # log attention_maps of agent0
                        for agent_id in range(len(self.agents)):
                            # states[:, 0, self.visible_range//2, self.visible_range//2] = 1
                            state = F.interpolate(
                                states,
                                size=(self.visible_range * 20, self.visible_range * 20),
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
                            self.writer.add_image(
                                f"attention_{agent_id}/observation",
                                image,
                                self.global_step,
                                dataformats="CHW",
                            )

                    # log
                    self.writer.add_scalar(
                        "training/epsilon", torch.tensor(self.epsilon), self.global_step
                    )
                    self.writer.add_scalar(
                        "training/reward",
                        torch.tensor(rewards).mean(),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        "training/total_loss", total_loss_sum, self.global_step
                    )

                    self.global_step += 1
                    self.episode_step += 1

                self.episode_count += 1
                self.epsilon *= self.config.epsilon_decay
                self.epsilon = max(self.config.epsilon_end, self.epsilon)
                # self.epsilon = 0.05#max(0.05, 1.0 - (epoch+1)/7500)

                # update target network
                for agent in self.agents:
                    hard_update(agent.dqn_target, agent.dqn)

                self.log_scalars()
                self.log_heatmap()
                self.reset()

                # updates pbar
                pbar.set_description(f"[Step {self.global_step}]")
                pbar.set_postfix({"loss": total_loss_sum.item()})
                pbar.update(1)

        self.writer.close()
        pbar.close()

        for agent_id, agent in enumerate(self.agents):
            model_path = f"agent_{agent_id}.pth"
            torch.save(agent.dqn.to("cpu").state_dict(), model_path)
