from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from core.utils.buffer import ReplayBuffer
import numpy as np


class AbstractTrainer(ABC):
    def __init__(self, config: DictConfig, environment):
        self.config = config
        self.env = environment

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer = ReplayBuffer(config.capacity, state_conv=True)
        self.agents = self.generate_agents()
        self.order = np.arange(environment.num_agents)

        self.states = self.env.reset()
        self.global_step = 0
        self.episode_count = 0
        self.epsilon = config.epsilon_initial

    def populate(self, steps: int):
        with tqdm(total=steps) as pbar:
            pbar.set_description("Populating buffer")
            for i in range(steps):
                self.play_step(epsilon=1.0)
                pbar.update(1)
            pbar.close()

    def reset(self):
        self.states = self.env.reset()
        self.episode_reward = 0
        self.episode_step = 0

    @abstractmethod
    def generate_agents(self):
        raise NotImplementedError()

    @abstractmethod
    def loss_and_update(self, batch):
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        raise NotImplementedError()

    def setup(self):
        pass

    def endup(self):
        pass

    def training_epoch_start(self, epoch: int):
        pass

    def training_epoch_end(self):
        pass

    @abstractmethod
    def training_step(self, step: int):
        raise NotImplementedError()

    def run(self):
        self.setup()

        with tqdm(total=self.config.max_epochs) as pbar:
            for epoch in range(self.config.max_epochs):
                self.training_epoch_start(epoch)
                for step in range(self.config.max_episode_length):
                    self.total_loss_sum = 0.0
                    self.training_step(step)
                    self.global_step += 1
                    self.episode_step += 1
                self.training_epoch_end()
                self.episode_count += 1

                pbar.set_description(f"[Step {self.global_step}]")
                pbar.set_postfix({"loss": self.total_loss_sum.item()})
                pbar.update(1)

        pbar.close()
        self.endup()
