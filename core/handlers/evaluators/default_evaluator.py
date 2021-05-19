import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from torch.nn import functional as F
from torchvision.utils import make_grid

from .abstract_evaluator import AbstractEvaluator

sns.set()


class DefaultEvaluator(AbstractEvaluator):
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        for agent_id in self.order:
            # normalize states [0, map.SIZE] -> [0, 1.0]
            states = torch.tensor(self.states).float()

            action, _ = self.agents[agent_id].get_action(states[agent_id], epsilon)
            actions[agent_id] = action

        rewards, _, new_states = self.env.step(actions)

        self.states = new_states

        return states, rewards

    def setup(self):
        self.reset()

        # load brain networks and weights
        self.load_state_dict()

    def validation_step(self, step: int, epoch: int):
        # execute in environment
        states, rewards = self.play_step()
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        if epoch % (self.config.max_epochs // 10) == 0 and step == (
            self.config.max_episode_length // 2
        ):
            for agent_id in range(self.env.num_agents):
                image = self.env.observation_handler.render(states[agent_id])

                wandb.log(
                    {
                        f"agent_{str(agent_id)}/observation": [
                            wandb.Image(
                                data_or_path=image,
                                caption="local observation",
                            )
                        ]
                    },
                    step=self.global_step,
                )

        wandb.log(
            {"training_step/total_reward": np.sum(rewards)},
            step=self.global_step,
        )

        for agent_id, reward in enumerate(rewards):
            wandb.log(
                {f"agent_{str(agent_id)}/step_reward": reward},
                step=self.global_step,
            )

    def validation_epoch_end(self):
        self.env.accumulate_heatmap()
        self.log_scalar()
        if self.episode_count % (self.config.max_epochs // 10) == 0:
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

    def log_heatmap2(self):
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

        for agent_id in range(self.env.num_agents):
            # log heatmap_agents
            fig = plt.figure()
            sns.heatmap(
                self.env.heatmap_accumulated_agents[agent_id].T,
                vmin=0,
                cmap="Blues",
                square=True,
                cbar_kws={"shrink": 0.65},  # 4agents:1.0, 8agents:0.65
                xticklabels=list(
                    str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
                ),
                yticklabels=list(
                    str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
                ),
            )
            plt.title(f"Agent {agent_id}")
            heatmap_accumulated_agents.append(
                wandb.Image(data_or_path=fig, caption=f"Agent {agent_id}")
            )
            # plt.close()

            # log heatmap_complete
            fig = plt.figure()
            sns.heatmap(
                self.env.heatmap_accumulated_complete[agent_id].T,
                vmin=0,
                cmap="Blues",
                square=True,
                cbar_kws={"shrink": 0.65},
                xticklabels=list(
                    str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
                ),
                yticklabels=list(
                    str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
                ),
            )
            plt.title(f"Agent {agent_id}")
            heatmap_accumulated_complete.append(
                wandb.Image(data_or_path=fig, caption=f"Agent {agent_id}")
            )
            # plt.close()

        # log heatmap_events
        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_accumulated_objects.T,
            vmin=0,
            cmap="Blues",
            square=True,
            cbar_kws={"shrink": 0.65},
            xticklabels=list(
                str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
            ),
            yticklabels=list(
                str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
            ),
        )
        heatmap_accumulated_objects.append(fig)
        # plt.close()

        # log heatmap_events_left
        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_accumulated_objects_left.T,
            vmin=0,
            cmap="Blues",
            square=True,
            cbar_kws={"shrink": 0.65},
            xticklabels=list(
                str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
            ),
            yticklabels=list(
                str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
            ),
        )
        heatmap_accumulated_objects_left.append(fig)
        # plt.close()

        # log heatmap_wall_collision
        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_accumulated_wall_collision.T,
            vmin=0,
            cmap="Blues",
            square=True,
            cbar_kws={"shrink": 0.65},
            xticklabels=list(
                str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
            ),
            yticklabels=list(
                str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
            ),
        )
        heatmap_accumulated_wall_collision.append(fig)
        # plt.close()

        # log heatmap_agents_collision
        fig = plt.figure()
        sns.heatmap(
            self.env.heatmap_accumulated_agents_collision.T,
            vmin=0,
            cmap="Blues",
            square=True,
            cbar_kws={"shrink": 0.65},
            xticklabels=list(
                str(x) if x % 2 == 0 else "" for x in range(-size_x, size_x)
            ),
            yticklabels=list(
                str(y) if y % 2 == 0 else "" for y in range(size_y, -size_y, -1)
            ),
        )
        heatmap_accumulated_agents_collision.append(fig)
        # plt.close()

        wandb.log(
            {
                "episode/heatmap_agents": heatmap_accumulated_agents,
                "episode/heatmap_complete": heatmap_accumulated_complete,
                "episode/heatmap_objects": heatmap_accumulated_objects,
                "episode/heatmap_objects_left": heatmap_accumulated_objects_left,
                "episode/heatmap_wall_collision": heatmap_accumulated_wall_collision,
                "episode/heatmap_agents_collision": heatmap_accumulated_agents_collision,
            },
            step=self.global_step - 1,
        )

    def log_heatmap(self):
        heatmap = torch.zeros(
            self.env.num_agents, 3, self.env.world.map.SIZE_X, self.env.world.map.SIZE_Y
        )

        for agent_id in range(self.env.num_agents):
            # add agent path information
            heatmap_agents = (
                0.5
                * self.env.heatmap_agents[agent_id, ...]
                / np.max(self.env.heatmap_agents[agent_id, ...])
            )
            heatmap_agents = np.where(
                heatmap_agents > 0, heatmap_agents + 0.5, heatmap_agents
            )
            heatmap[agent_id, 2, ...] += torch.from_numpy(heatmap_agents)

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
        heatmap = make_grid(heatmap, nrow=3)
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
