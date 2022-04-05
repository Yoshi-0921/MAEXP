import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from core.utils.buffer import Experience
from copy import deepcopy
from .default_trainer import DefaultTrainer

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
sns.set()


class AttentionWanderingTrainer(DefaultTrainer):
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states
        for agent_id in self.order:
            if self.config.agent_tasks[int(agent_id)] == -1:
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue

            action, attns = self.agents[agent_id].get_action_attns(
                deepcopy(states[agent_id]), epsilon
            )
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        rewards, dones, new_states = self.env.step(actions, self.order)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(deepcopy(exp))

        del exp

        self.states = new_states

        return states, rewards, attention_maps

    def loop_step(self, step: int, epoch: int):
        # train based on experiments
        for batch in self.dataloader:
            loss_list = self.loss_and_update(batch)
            loss_list = self.loss_and_update(batch)
            self.step_loss_sum += sum(
                [loss_dict["total_loss"].item() for loss_dict in loss_list]
            )

        # execute in environment
        states, rewards, attention_maps = self.play_step(self.epsilon)
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        if epoch % (self.max_epochs // 10) == 0 and step == (
            self.max_episode_length // 2
        ):
            for agent_id, agent in enumerate(self.agents):
                if self.config.agent_tasks[int(agent_id)] == -1:
                    continue

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

                for attention_map, (view_method, image) in zip(
                    attention_maps[agent_id], images.items()
                ):
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
