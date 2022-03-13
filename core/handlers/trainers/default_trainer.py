import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from core.utils.color import RGB_COLORS
from torch.nn import functional as F
from torchvision.utils import make_grid

from .abstract_trainer import AbstractTrainer

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
sns.set()


class DefaultTrainer(AbstractTrainer):
    def loop_epoch_start(self, epoch: int):
        if epoch == (self.max_epochs // 2):
            self.save_state_dict(epoch=epoch)

    def loop_step(self, step: int, epoch: int):
        # train based on experiments
        for batch in self.dataloader:
            loss_list = self.loss_and_update(batch)
            self.total_loss_sum += torch.sum(loss_list).item()
            self.total_loss_agents += loss_list

        if (
            self.config.destination_channel
            and self.episode_step % self.config.reset_destination_period == 0
        ):
            self.env.world.map.reset_destination_area()

        # execute in environment
        states, rewards = self.play_step(self.epsilon)
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        if epoch % (self.max_epochs // 4) == 0 and step == (
            self.max_episode_length // 2
        ):
            # log attention_maps of agent0
            for agent_id in range(len(self.agents)):
                if self.config.destination_channel:
                    fig = plt.figure()
                    sns.heatmap(
                        self.env.world.map.destination_area_matrix[agent_id].T,
                        square=True,
                    )

                    wandb.log(
                        {
                            f"agent_{str(agent_id)}/destination_channel": [
                                wandb.Image(
                                    data_or_path=fig,
                                    caption="destination channel",
                                )
                            ]
                        },
                        step=self.global_step,
                    )

                images = self.env.observation_handler.render(states[agent_id])
                for view_method, image in images.items():
                    wandb.log(
                        {
                            f"agent_{str(agent_id)}/{view_method}_observation": [
                                wandb.Image(
                                    data_or_path=image,
                                    caption=f"{view_method} observation",
                                )
                            ]
                        },
                        step=self.global_step,
                    )

        wandb.log(
            {
                "training/epsilon": self.epsilon,
                "training/total_reward": np.sum(rewards),
                "training/total_loss": self.total_loss_sum,
            },
            step=self.global_step,
        )

        for agent_id, (loss, reward) in enumerate(zip(self.total_loss_agents, rewards)):
            wandb.log(
                {
                    f"agent_{str(agent_id)}/step_loss": loss,
                    f"agent_{str(agent_id)}/step_reward": reward,
                },
                step=self.global_step,
            )

    def loop_epoch_end(self):
        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.config.epsilon_end, self.epsilon)

        if self.episode_count % self.synchronize_frequency == 0:
            for agent in self.agents:
                agent.synchronize_brain()

        self.log_scalar()
        if self.episode_count % (self.max_epochs // 10) == 0:
            self.log_heatmap()
        self.reset()

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

    def log_heatmap(self):
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
