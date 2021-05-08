import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from core.utils.buffer import Experience
from core.utils.dataset import RLDataset
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .abstract_trainer import AbstractTrainer

sns.set()


class MATTrainer(AbstractTrainer):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config=config, environment=environment)
        self.visible_range = self.config.visible_range
        wandb.init(
            project="MAEXP", entity="yoshi-0921", name=config.name, config=dict(config)
        )

    def loss_and_update(self, batch):
        loss_list = []
        states, actions, rewards, dones, next_states = batch
        for agent_id, agent in enumerate(self.agents):
            loss = agent.learn(
                states[agent_id],
                actions[agent_id],
                rewards[agent_id],
                dones[agent_id],
                next_states[agent_id],
            )
            loss_list.append(loss)

        return torch.from_numpy(np.array(loss_list, dtype=np.float))

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        for agent_id in self.order:
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action, attns = self.agents[agent_id].get_action(states[agent_id], epsilon)
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        rewards, dones, new_states = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return states, rewards, attention_maps

    def setup(self):
        # set dataloader
        dataset = RLDataset(self.buffer, self.config.batch_size)
        self.dataloader = DataLoader(
            dataset=dataset, batch_size=self.config.batch_size, pin_memory=True
        )

        # populate buffer
        self.populate(self.config.populate_steps)
        self.reset()

    def training_step(self, step: int, epoch: int):
        # train based on experiments
        for batch in self.dataloader:
            loss_list = self.loss_and_update(batch)
            self.total_loss_sum += torch.sum(loss_list)
            self.total_loss_agents += loss_list

        # execute in environment
        states, rewards, attention_maps = self.play_step(self.epsilon)
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        if epoch % 10 == 0 and step % (self.config.max_episode_length // 5) == 0:
            # log attention_maps of agent0
            for agent_id, agent in enumerate(self.agents):
                attention_map = (
                    attention_maps[agent_id]
                    .mean(dim=0)[0, :, 0, 1:]
                    .view(-1, agent.patched_size_x, agent.patched_size_y)
                    .cpu()
                    .detach()
                )

                fig = plt.figure()
                sns.heatmap(
                    torch.t(attention_map.mean(dim=0)),
                    vmin=0,
                    square=True,
                    annot=True,
                    fmt=".3f",
                    vmax=0.25,
                )
                wandb.log(
                    {
                        f"attentions/agent_{str(agent_id)}/heatmap": [
                            wandb.Image(
                                data_or_path=fig,
                                caption="mean attention heatmap",
                            )
                        ]
                    },
                    step=self.global_step
                )
                plt.close()

                for head_id, am in enumerate(attention_map):
                    fig = plt.figure()
                    sns.heatmap(
                        torch.t(am), vmin=0, square=True, annot=True, fmt=".3f", vmax=0.25
                    )
                    wandb.log(
                        {
                            f"attentions/agent_{str(agent_id)}/heatmap_{str(head_id)}": [
                                wandb.Image(
                                    data_or_path=fig,
                                    caption=f"attention heatmap from head {str(head_id)}",
                                )
                            ]
                        },
                        step=self.global_step
                    )
                    plt.close()

                state = torch.t(states[agent_id])
                image = np.zeros(
                    (self.visible_range, self.visible_range, 3),
                    dtype=np.float,
                )
                obs = state.permute(0, 2, 1).numpy() * 255.0

                # add agent information (Blue)
                image[..., 0] += obs[0]
                # add object information (Yellow)
                image[..., 1] += obs[1]
                image[..., 2] += obs[1]
                # add invisible area information (White)
                image[..., 0] -= obs[2]
                image[..., 1] -= obs[2]
                image[..., 2] -= obs[2]

                wandb.log(
                    {
                        f"attentions/agent_{str(agent_id)}/observation": [
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
                "training/epsilon": self.epsilon,
                "training/total_reward": np.sum(rewards),
                "training/total_loss": self.total_loss_sum,
            },
            step=self.global_step,
        )

        for agent_id, (loss, reward) in enumerate(zip(self.total_loss_agents, rewards)):
            wandb.log(
                {
                    f"training/agents/loss_{str(agent_id)}": loss,
                    f"training/agents/reward_{str(agent_id)}": reward,
                },
                step=self.global_step,
            )

    def training_epoch_end(self):
        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.config.epsilon_end, self.epsilon)

        for agent in self.agents:
            agent.synchronize_brain()

        self.log_scalar()
        self.log_heatmap()
        self.reset()

    def log_scalar(self):
        wandb.log(
            {
                "episode/episode_reward": self.episode_reward_sum,
                "episode/episode_step": self.episode_step,
                "episode/global_step": self.global_step,
                "env/objects_left": self.env.objects_generated
                - self.env.objects_completed,
                "env/objects_completed": self.env.objects_completed,
                "env/agents_collided": self.env.agents_collided,
                "env/walls_collided": self.env.walls_collided,
            },
            step=self.global_step - 1,
        )

        for agent_id, reward in enumerate(self.episode_reward_agents):
            wandb.log(
                {
                    f"episode/agents/episode_reward_{str(agent_id)}": reward,
                },
                step=self.global_step - 1,
            )

    def log_heatmap(self):
        heatmap = torch.zeros(
            self.env.num_agents, 3, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y
        )

        for i in range(self.env.num_agents):
            # add agent path information
            heatmap_agents = (
                0.5
                * self.env.heatmap_agents[i, ...]
                / np.max(self.env.heatmap_agents[i, ...])
            )
            heatmap_agents = np.where(
                heatmap_agents > 0, heatmap_agents + 0.5, heatmap_agents
            )
            heatmap[i, 2, ...] += torch.from_numpy(heatmap_agents)

        # add wall information
        heatmap[:, :, ...] += torch.from_numpy(self.env.world.map.wall_matrix)

        # add objects information
        heatmap_objects = (
            0.8 * self.env.heatmap_objects / np.max(self.env.heatmap_objects)
        )
        heatmap_objects = np.where(
            heatmap_objects > 0, heatmap_objects + 0.2, heatmap_objects
        )
        heatmap[:, torch.tensor([0, 1]), ...] += torch.from_numpy(heatmap_objects)

        heatmap = F.interpolate(
            heatmap,
            size=(self.env.world.map.SIZE_X * 10, self.env.world.map.SIZE_Y * 10),
        )
        heatmap = torch.transpose(heatmap, 2, 3)
        heatmap = make_grid(heatmap, nrow=2)
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
