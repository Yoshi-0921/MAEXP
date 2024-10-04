import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from omegaconf import DictConfig
from thop import clever_format, profile
from torch.nn import functional as F
from torchinfo import summary
from torchvision.utils import make_grid
from tqdm import tqdm

from core.agents import generate_agents
from core.utils.color import RGB_COLORS
from core.utils.logging import initialize_logging

logger = initialize_logging(__name__)


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

    def log_heatmap(self):
        self.log_episode_trajectory()
        self.log_accumulated_heatmap()
        for agent_id in range(len(self.agents)):
            self.log_observation_channels(agent_id)
        self.log_observation_channels_correlation()
        self.log_observation_agents_correlation()

    def log_episode_trajectory(self):
        heatmap = torch.zeros(
            self.env.num_agents, 3, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y
        )

        for agent_id, color in enumerate(self.config.agents_color):
            # add agent path information
            heatmap_agents = (
                0.5
                * self.env.heatmap_agents[agent_id, ...]
                / max(np.max(self.env.heatmap_agents[agent_id, ...]), 1)
            )
            heatmap_agents = np.where(
                heatmap_agents > 0, heatmap_agents + 0.5, heatmap_agents
            )
            rgb = RGB_COLORS[color]
            rgb = np.expand_dims(np.asarray(rgb), axis=(1, 2))
            heatmap[agent_id] += torch.from_numpy(heatmap_agents) * rgb

        # add wall information
        heatmap[:, :, ...] += torch.from_numpy(self.env.world.map.wall_matrix)

        # add objects information
        heatmap_objects = (
            0.8 * self.env.heatmap_objects / np.max(self.env.heatmap_objects)
        )
        heatmap_objects = np.where(
            heatmap_objects > 0, heatmap_objects + 0.2, heatmap_objects
        )
        for heatmap_object, color in zip(heatmap_objects, self.config.objects_color):
            rgb = RGB_COLORS[color]
            rgb = np.expand_dims(np.asarray(rgb), axis=(1, 2))
            heatmap[:, ...] += torch.from_numpy(heatmap_object) * rgb

        heatmap = F.interpolate(
            heatmap,
            size=(self.env.world.map.SIZE_X * 10, self.env.world.map.SIZE_Y * 10),
        )
        heatmap = torch.transpose(heatmap, 2, 3)
        heatmap = make_grid(heatmap, nrow=self.config.episode_hm_nrow)
        wandb.log(
            {
                "episode/heatmap": [
                    wandb.Image(
                        data_or_path=heatmap,
                        caption="Episode heatmap",
                    )
                ]
            },
            step=self.global_step - 1,
        )

    def log_accumulated_heatmap(self):
        (
            heatmap_accumulated_agents,
            heatmap_accumulated_complete,
            heatmap_accumulated_objects,
            heatmap_accumulated_objects_left,
            heatmap_accumulated_wall_collision,
            heatmap_accumulated_agents_collision,
        ) = ([], [], [], [], [], [])
        size_x = self.env.world.map.SIZE_X // 2
        size_y = self.env.world.map.SIZE_Y // 2

        if self.config.save_pdf_figs:
            if not os.path.exists("heatmap_accumulated_agents"):
                os.mkdir("heatmap_accumulated_agents")
            if not os.path.exists("heatmap_accumulated_complete"):
                os.mkdir("heatmap_accumulated_complete")
            os.mkdir(f"heatmap_accumulated_agents/epoch_{self.episode_count}")
            os.mkdir(f"heatmap_accumulated_complete/epoch_{self.episode_count}")

        for agent_id in range(self.env.num_agents):
            # log heatmap_agents
            fig = plt.figure()
            sns.heatmap(
                self.env.heatmap_accumulated_agents[agent_id].T,
                vmin=0,
                cmap="PuBu",
                square=True,
                cbar_kws={"shrink": 0.61},
                xticklabels=list(
                    str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
                ),
                yticklabels=list(
                    str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
                ),
            )
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if self.config.save_pdf_figs:
                plt.savefig(f'heatmap_accumulated_agents/epoch_{self.episode_count}/agent_{str(agent_id)}.pdf', dpi=300)

            heatmap_accumulated_agents.append(
                wandb.Image(data_or_path=fig, caption=f"Agent {agent_id}")
            )
            plt.close()

            # log heatmap_complete
            fig = plt.figure()
            sns.heatmap(
                self.env.heatmap_accumulated_complete[agent_id].T,
                vmin=0,
                cmap="PuBu",
                square=True,
                cbar_kws={"shrink": 0.61},
                xticklabels=list(
                    str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
                ),
                yticklabels=list(
                    str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
                ),
            )
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if self.config.save_pdf_figs:
                plt.savefig(f'heatmap_accumulated_complete/epoch_{self.episode_count}/agent_{str(agent_id)}.pdf', dpi=300)

            heatmap_accumulated_complete.append(
                wandb.Image(data_or_path=fig, caption=f"Agent {agent_id}")
            )
            # plt.savefig(f"agent{agent_id}.pdf")
            plt.close()

        # log heatmap_events
        for object_type, heatmap in enumerate(heatmap_accumulated_objects):
            fig = plt.figure()
            sns.heatmap(
                heatmap.T,
                vmin=0,
                cmap="Blues",
                square=True,
                xticklabels=list(
                    str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
                ),
                yticklabels=list(
                    str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
                ),
            )
            heatmap_accumulated_objects.append(
                wandb.Image(data_or_path=fig, caption=f"Object {object_type} generated")
            )
            plt.close()

        for object_type, heatmap in enumerate(self.env.heatmap_accumulated_objects_left):
            fig = plt.figure()
            sns.heatmap(
                heatmap.T,
                vmin=0,
                cmap="Blues",
                square=True,
                xticklabels=list(
                    str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
                ),
                yticklabels=list(
                    str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
                ),
            )
            heatmap_accumulated_objects_left.append(
                wandb.Image(data_or_path=fig, caption=f"Object {object_type} left")
            )
            plt.close()

        # log heatmap_wall_collision
        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_accumulated_wall_collision.T,
            vmin=0,
            cmap="Blues",
            square=True,
            xticklabels=list(
                str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
            ),
            yticklabels=list(
                str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
            ),
        )
        heatmap_accumulated_wall_collision.append(
            wandb.Image(data_or_path=fig, caption="Wall collision")
        )
        plt.close()

        # log heatmap_agents_collision
        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_accumulated_agents_collision.T,
            vmin=0,
            cmap="Blues",
            square=True,
            xticklabels=list(
                str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
            ),
            yticklabels=list(
                str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
            ),
        )
        heatmap_accumulated_agents_collision.append(
            wandb.Image(data_or_path=fig, caption="Agents collision")
        )
        plt.close()

        wandb.log(
            {
                "heatmaps/agents_path": heatmap_accumulated_agents,
                "heatmaps/objects_completion": heatmap_accumulated_complete,
                "heatmaps/objects_generated": heatmap_accumulated_objects,
                "heatmaps/objects_left": heatmap_accumulated_objects_left,
                "heatmaps/wall_collision": heatmap_accumulated_wall_collision,
                "heatmaps/agents_collision": heatmap_accumulated_agents_collision,
            },
            step=self.global_step - 1,
        )

    def log_scalar(self):
        wandb.log(
            {
                "episode/episode_reward": self.episode_reward_sum,
                "episode/episode_step": self.episode_step,
                "episode/global_step": self.global_step,
                "episode/objects_left": self.env.objects_generated
                - self.env.objects_completed,
                "episode/objects_completed": self.env.objects_completed,
                "episode/agents_collided": self.env.agents_collided,
                "episode/walls_collided": self.env.walls_collided,
            },
            step=self.global_step - 1,
        )

        for agent_id, reward in enumerate(self.episode_reward_agents):
            wandb.log(
                {
                    f"agent_{str(agent_id)}/episode_reward": reward,
                },
                step=self.global_step - 1,
            )

    def log_destination_channel(self, agent_id: int):
        if self.config.destination_channel:
            fig = plt.figure()
            sns.heatmap(
                self.env.world.map.destination_area_matrix[agent_id].T,
                square=True,
            )
            if self.config.save_pdf_figs:
                plt.savefig(f'agent_{str(agent_id)}/step_{self.global_step}/destination_channel.pdf', dpi=300)

            wandb.log(
                {
                    f"agent_{str(agent_id)}/destination_channel": [
                        wandb.Image(
                            data_or_path=fig,
                            caption="destination channel",
                        )
                    ]
                },
                step=self.global_step-1,
            )

    def log_observation_channels(self, agent_id: int):
        for ch_id, heatmap in enumerate(self.env.heatmap_accumulated_observation[agent_id]):
            if ch_id == self.env.observation_handler.get_channel - 1:
                heatmap = heatmap * -1
            fig = plt.figure()
            sns.heatmap(
                heatmap.T,
                square=True,
                annot=True,
                fmt=".4g",
                annot_kws={'fontsize': 6}
            )

            wandb.log(
                {
                    f"agent_{str(agent_id)}_observation/channel_{str(ch_id)}": [
                        wandb.Image(
                            data_or_path=fig,
                            caption=f"observation channel ({str(ch_id)})",
                        )
                    ]
                },
                step=self.global_step-1,
            )
            logger.info(f"agent_{str(agent_id)}_observation/channel_{str(ch_id)}:")
            logger.info(heatmap)

    def log_observation_channels_correlation(self):
        channels_correlation = np.zeros(
            shape=(self.env.num_agents, self.env.observation_handler.get_channel),
            dtype=np.int32
        )

        for agent_id, heatmap_accumulated_observation in enumerate(self.env.heatmap_accumulated_observation):
            for ch_id, heatmap in enumerate(heatmap_accumulated_observation):
                if ch_id == self.env.observation_handler.get_channel - 1:
                    heatmap = heatmap * -1
                channels_correlation[agent_id, ch_id] = np.sum(heatmap)

        fig = plt.figure()
        sns.heatmap(
            channels_correlation,
            square=True,
            annot=True,
            fmt=".4g",
            annot_kws={'fontsize': 6}
        )

        wandb.log(
            {
                f"agents_observation/channels_correlation": [
                    wandb.Image(
                        data_or_path=fig,
                        caption=f"observation channels correlation",
                    )
                ]
            },
            step=self.global_step-1,
        )
        logger.info(f"agents_observation/channels_correlation:")
        logger.info(channels_correlation)

    def log_observation_agents_correlation(self):
        agents_correlation = np.zeros(
            shape=(self.env.num_agents, self.env.num_agents),
            dtype=np.int32
        )

        for agent_id, heatmap_accumulated_observation in enumerate(self.env.heatmap_accumulated_observation):
            for ch_id, heatmap in enumerate(heatmap_accumulated_observation[:self.env.num_agents]):
                agents_correlation[agent_id, ch_id] = np.sum(heatmap)

        fig = plt.figure()
        sns.heatmap(
            agents_correlation,
            square=True,
            annot=True,
            fmt=".4g",
            annot_kws={'fontsize': 6}
        )

        wandb.log(
            {
                f"agents_observation/agents_correlation": [
                    wandb.Image(
                        data_or_path=fig,
                        caption=f"observation agents correlation",
                    )
                ]
            },
            step=self.global_step-1,
        )
        logger.info(f"agents_observation/agents_correlation:")
        logger.info(agents_correlation)
