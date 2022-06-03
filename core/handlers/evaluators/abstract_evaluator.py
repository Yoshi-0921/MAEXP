import random
from abc import ABC

import torch
import wandb
from omegaconf import DictConfig

from ..abstract_loop_handler import AbstractLoopHandler


class AbstractEvaluator(AbstractLoopHandler, ABC):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config=config, environment=environment)
        self.max_epochs = config.validate_epochs

        wandb.init(
            project=config.project_name + "-evaluation",
            entity="yoshi-0921",
            name=config.name + "_evaluation",
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
                config.observation_area_mask + "_area_mask",
                config.agent_view_method + "_agent_view_method",
                config.object_view_method + "_object_view_method",
            ],
        )

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.00):
        actions = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        for agent_id in self.order:
            if self.config.agent_tasks[int(agent_id)] == "-1":
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue

            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action = self.agents[agent_id].get_action(states[agent_id], epsilon)
            actions[agent_id] = action

        rewards, _, new_states = self.env.step(actions, self.order)

        self.states = new_states

        return states, rewards

    def setup(self):
        self.reset()
        # load brain networks and weights
        self.load_state_dict()
