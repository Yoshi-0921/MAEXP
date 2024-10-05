import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb

from core.utils.buffer import Experience
from core.utils.logging import initialize_logging

from .default_trainer import DefaultTrainer

# plt.use('Agg')
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
sns.set()

logger = initialize_logging(__name__)


class AttentionTrainer(DefaultTrainer):
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states
        for agent_id in self.order:
            if self.config.agent_tasks[int(agent_id)] == "-1":
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue

            action, attns = self.agents[agent_id].get_action_attns(
                deepcopy(states[agent_id]), epsilon
            )
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        rewards, dones, new_states = self.env.step(actions, self.order)

        exp = Experience(states, actions, rewards, dones, new_states)

        self.buffer.append(deepcopy(exp))

        del exp

        self.states = new_states

        return states, rewards, dones, attention_maps

    def loop_step(self, step: int, epoch: int) -> bool:
        # train based on experiments
        for batch in self.dataloader:
            loss_list = self.loss_and_update(batch)
            self.step_loss_sum += sum(
                [loss_dict["total_loss"].item() for loss_dict in loss_list]
            )

        if (
            self.config.reset_destination_period
            and self.episode_step % self.config.reset_destination_period == 0
        ):
            self.env.world.map.reset_destination_area()

        # execute in environment
        states, rewards, dones, attention_maps = self.play_step(self.epsilon)
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        if epoch % (self.max_epochs // 10) == 0 and step == (
            self.max_episode_length // 2
        ):
            self.save_state_dict(epoch=epoch)
            for agent_id, agent in enumerate(self.agents):
                if self.config.agent_tasks[int(agent_id)] == "-1":
                    continue

                self.log_destination_channel(agent_id)

                images = self.env.observation_handler.render(states[agent_id])

                if self.config.destination_channel:
                    images["destination"] = None

                for attention_map, (view_method, image) in zip(
                    attention_maps[agent_id], images.items()
                ):
                    if view_method == "destination":
                        attention_map = (
                            attention_map.mean(dim=0)[0, :, 0, 1:]
                            .view(
                                -1,
                                getattr(agent.brain, "relative_patched_size_x"),
                                getattr(agent.brain, "relative_patched_size_y"),
                            )
                            .cpu()
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
                                f"agent_{str(agent_id)}/{view_method}_attention_mean": [
                                    wandb.Image(
                                        data_or_path=fig,
                                        caption=f"mean {view_method} attention heatmap",
                                    )
                                ]
                            },
                            step=self.global_step,
                        )
                        continue

                    # TODO Remove this code - cda3 exp
                    if False:
                        pos_attention_map = (
                            attention_map.mean(dim=0)[0, :, 0, 2:]
                            .view(
                                -1,
                                getattr(agent.brain, f"{view_method}_patched_size_x"),
                                getattr(agent.brain, f"{view_method}_patched_size_y"),
                            )
                            .cpu()
                        )

                        fig = plt.figure()
                        sns.heatmap(
                            torch.t(pos_attention_map.mean(dim=0)),
                            vmin=0,
                            square=True,
                            annot=True,
                            fmt=".3f",
                            vmax=0.25,
                        )

                        wandb.log(
                            {
                                f"agent_{str(agent_id)}/{view_method}_pos_attention_mean": [
                                    wandb.Image(
                                        data_or_path=fig,
                                        caption=f"mean {view_method} pos attention heatmap",
                                    )
                                ]
                            },
                            step=self.global_step,
                        )

                        fig_list = []
                        for head_id, am in enumerate(pos_attention_map):
                            fig = plt.figure()
                            sns.heatmap(
                                torch.t(am),
                                vmin=0,
                                square=True,
                                annot=True,
                                fmt=".3f",
                                vmax=0.25,
                            )
                            fig_list.append(
                                wandb.Image(
                                    data_or_path=fig,
                                    caption=f"{view_method} pos attention heatmap from head {str(head_id)}",
                                )
                            )

                        wandb.log(
                            {
                                f"agent_{str(agent_id)}/{view_method}_pos_attention_heads": fig_list
                            },
                            step=self.global_step,
                        )

                        neg_attention_map = (
                            attention_map.mean(dim=0)[0, :, 1, 2:]
                            .view(
                                -1,
                                getattr(agent.brain, f"{view_method}_patched_size_x"),
                                getattr(agent.brain, f"{view_method}_patched_size_y"),
                            )
                            .cpu()
                        )

                        fig = plt.figure()
                        sns.heatmap(
                            torch.t(neg_attention_map.mean(dim=0)),
                            vmin=0,
                            square=True,
                            annot=True,
                            fmt=".3f",
                            vmax=0.25,
                        )

                        wandb.log(
                            {
                                f"agent_{str(agent_id)}/{view_method}_neg_attention_mean": [
                                    wandb.Image(
                                        data_or_path=fig,
                                        caption=f"mean {view_method} neg attention heatmap",
                                    )
                                ]
                            },
                            step=self.global_step,
                        )

                        fig_list = []
                        for head_id, am in enumerate(neg_attention_map):
                            fig = plt.figure()
                            sns.heatmap(
                                torch.t(am),
                                vmin=0,
                                square=True,
                                annot=True,
                                fmt=".3f",
                                vmax=0.25,
                            )
                            fig_list.append(
                                wandb.Image(
                                    data_or_path=fig,
                                    caption=f"{view_method} neg attention heatmap from head {str(head_id)}",
                                )
                            )

                        wandb.log(
                            {
                                f"agent_{str(agent_id)}/{view_method}_neg_attention_heads": fig_list
                            },
                            step=self.global_step,
                        )

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
                        continue

                    attention_map = (
                        attention_map.mean(dim=0)[0, :, 0, 1:]
                        .view(
                            -1,
                            getattr(agent.brain, f"{view_method}_patched_size_x"),
                            getattr(agent.brain, f"{view_method}_patched_size_y"),
                        )
                        .cpu()
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
                            f"agent_{str(agent_id)}/{view_method}_attention_mean": [
                                wandb.Image(
                                    data_or_path=fig,
                                    caption=f"mean {view_method} attention heatmap",
                                )
                            ]
                        },
                        step=self.global_step,
                    )

                    fig_list = []
                    for head_id, am in enumerate(attention_map):
                        fig = plt.figure()
                        sns.heatmap(
                            torch.t(am),
                            vmin=0,
                            square=True,
                            annot=True,
                            fmt=".3f",
                            vmax=0.25,
                        )
                        fig_list.append(
                            wandb.Image(
                                data_or_path=fig,
                                caption=f"{view_method} attention heatmap from head {str(head_id)}",
                            )
                        )

                    wandb.log(
                        {
                            f"agent_{str(agent_id)}/{view_method}_attention_heads": fig_list
                        },
                        step=self.global_step,
                    )

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
                "training_step/epsilon": self.epsilon,
                "training_step/total_reward": np.sum(rewards),
                "training_step/total_loss": self.step_loss_sum,
            },
            step=self.global_step,
        )

        for agent_id, (loss_dict, reward) in enumerate(zip(loss_list, rewards)):
            output_dict = {
                f"agent_{str(agent_id)}/step_{loss_name}": loss
                for loss_name, loss in loss_dict.items()
            }
            output_dict.update(
                {
                    f"agent_{str(agent_id)}/step_reward": reward,
                }
            )
            wandb.log(
                output_dict,
                step=self.global_step,
            )

        return dones[0]
