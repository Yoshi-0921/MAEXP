from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import wandb
from core.agents import generate_agents
from omegaconf import DictConfig
from thop import clever_format, profile
from torchinfo import summary
from tqdm import tqdm


class AbstractLoopHandler(ABC):
    def __init__(self, config: DictConfig, environment):
        self.config = config
        self.env = environment

        self.agents = generate_agents(
            config=config,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )
        self.order = np.arange(environment.num_agents)

        self.reset()
        self.env.render_world()
        self.global_step = 0
        self.episode_count = 0
        self.epsilon = 0

        self.max_epochs = config.max_epochs
        self.max_episode_length = config.max_episode_length

    def reset(self):
        self.states = self.env.reset()
        self.episode_reward_sum = 0.0
        self.episode_reward_agents = np.zeros(self.env.num_agents)
        self.episode_step = 0

    def populate(self, steps: int):
        with tqdm(total=steps) as pbar:
            pbar.set_description("Populating buffer")
            for _ in range(steps):
                self.play_step(epsilon=1.0)
                pbar.update(1)
            pbar.close()

    @abstractmethod
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        raise NotImplementedError()

    def setup(self):
        pass

    def endup(self):
        pass

    def loop_epoch_start(self, epoch: int):
        pass

    def loop_epoch_end(self):
        pass

    def loop_step(self, step: int) -> bool:
        raise NotImplementedError()

    def run(self):
        self.setup()

        with tqdm(total=self.max_epochs) as pbar:
            for epoch in range(self.max_epochs):
                self.loop_epoch_start(epoch)
                for step in range(self.max_episode_length):
                    self.step_loss_sum = 0.0
                    self.step_loss_agents = torch.zeros(self.env.num_agents)
                    done = self.loop_step(step, epoch)
                    self.global_step += 1
                    self.episode_step += 1
                    if done:
                        break
                self.loop_epoch_end()
                self.episode_count += 1

                pbar.set_description(f"[Step {self.global_step}]")
                pbar.set_postfix({"loss": self.step_loss_sum})
                pbar.update(1)

        self.endup()

        pbar.close()

    def log_models(self):
        network_table = wandb.Table(columns=["Agent", "MACs", "Parameters"])
        model_artifact = wandb.Artifact(
            name=wandb.run.id + ".onnx",
            type="model_topology",
            metadata=dict(self.config),
        )

        for agent_id, agent in enumerate(self.agents):
            print(f"Agent {str(agent_id)}:")
            summary(model=agent.brain.network)
            dummy_input = {}
            for state_key, state_value in self.states[agent_id].items():
                if isinstance(state_value, torch.Tensor):
                    dummy_input[state_key] = torch.randn(
                        1, *state_value.shape, device=agent.brain.device
                    )
                else:
                    dummy_input[state_key] = state_value

            macs, params = clever_format(
                [*profile(agent.brain.network, inputs=(dummy_input,), verbose=False)],
                "%.3f",
            )
            network_table.add_data(
                f"Agent {str(agent_id)}", f"{str(macs)}", f"{str(params)}"
            )

            wandb.watch(
                models=agent.brain.network,
                log="all",
                log_freq=self.config.max_episode_length * (self.config.max_epochs // 5),
                idx=agent_id,
            )
            try:
                torch.onnx.export(
                    agent.brain.network, dummy_input, f"agent_{str(agent_id)}.onnx"
                )
                model_artifact.add_file(f"agent_{str(agent_id)}.onnx")
            except RuntimeError:
                pass

        wandb.log({"tables/Network description": network_table}, step=0)
        wandb.log_artifact(model_artifact)

    def save_state_dict(self, epoch: int = None):
        meta_config = dict(self.config)
        meta_config.update(
            {"save_state_epoch": epoch}
            if epoch
            else {"save_state_epoch": self.config.max_epochs - 1}
        )

        weight_artifact = wandb.Artifact(
            name=wandb.run.id + ".pth", type="pretrained_weight", metadata=meta_config
        )
        for agent_id, agent in enumerate(self.agents):
            model_path = f"agent{agent_id}.pth"
            torch.save(agent.brain.network.to("cpu").state_dict(), model_path)
            weight_artifact.add_file(model_path)
            agent.brain.network.to(agent.brain.device)

        wandb.log_artifact(weight_artifact)

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
