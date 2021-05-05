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

    @abstractmethod
    def run(self):
        raise NotImplementedError()
