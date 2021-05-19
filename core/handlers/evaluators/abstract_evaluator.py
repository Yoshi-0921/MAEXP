from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
from core.agents import generate_agents
from omegaconf import DictConfig
from tqdm import tqdm
from collections import OrderedDict


class AbstractEvaluator(ABC):
    def __init__(self, config: DictConfig, environment):
        self.config = config
        self.env = environment

        self.agents = generate_agents(
            config=config,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )
        self.order = np.arange(environment.num_agents)

        self.states = self.env.reset()
        self.env.render_world()
        self.global_step = 0
        self.episode_count = 0
        self.epsilon = 0

        wandb.init(
            project="MAEXP",
            entity="yoshi-0921",
            name=config.name + "_test",
            config=dict(config),
            tags=[
                config.world + "_world",
                config.environment + "_environment",
                config.agent_type + "_agent",
                config.brain + "_brain",
                config.phase + "_pahse",
                config.evaluator + "_evaluator",
                config.model.name + "_model",
                config.map.name + "_map",
            ],
        )

    def reset(self):
        self.states = self.env.reset()
        self.episode_reward_sum = 0.0
        self.episode_reward_agents = np.zeros(self.env.num_agents)
        self.episode_step = 0

    @abstractmethod
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        raise NotImplementedError()

    def setup(self):
        pass

    def endup(self):
        pass

    def validation_epoch_start(self, epoch: int):
        pass

    def validation_epoch_end(self):
        pass

    @abstractmethod
    def validation_step(self, step: int):
        raise NotImplementedError()

    def run(self):
        self.setup()

        with tqdm(total=self.config.validate_epochs) as pbar:
            for epoch in range(self.config.validate_epochs):
                self.validation_epoch_start(epoch)
                for step in range(self.config.max_episode_length):
                    self.total_loss_sum = 0.0
                    self.total_loss_agents = torch.zeros(self.env.num_agents)
                    self.validation_step(step, epoch)
                    self.global_step += 1
                    self.episode_step += 1
                self.validation_epoch_end()
                self.episode_count += 1

                pbar.set_description(f"[Step {self.global_step}]")
                pbar.update(1)

        self.endup()

        pbar.close()

    def load_state_dict(self):
        weight_artifact = wandb.use_artifact(
            self.config.pretrained_weight_path, type="pretrained_weight"
        )
        weight_artifact_dir = weight_artifact.download()
        for agent_id, agent in enumerate(self.agents):
            new_state_dict = OrderedDict()
            state_dict = torch.load(weight_artifact_dir + f"/agent{agent_id}.pth")
            for key, value in state_dict.items():
                if key in agent.brain.network.state_dict().keys():
                    new_state_dict[key] = value

            agent.brain.network.load_state_dict(new_state_dict)
