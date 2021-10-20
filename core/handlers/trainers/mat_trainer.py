import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from core.utils.buffer import Experience

from .default_trainer import DefaultTrainer

sns.set()


class MATTrainer(DefaultTrainer):
    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states
        for agent_id in self.order:
            action, attns = self.agents[agent_id].get_action_attns(
                states[agent_id], epsilon
            )
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        rewards, dones, new_states = self.env.step(actions)

        exp = Experience(self.states, actions, rewards, dones, new_states)

        self.buffer.append(exp)

        self.states = new_states

        return states, rewards, attention_maps

    def loop_step(self, step: int, epoch: int):
        # train based on experiments
        for batch in self.dataloader:
            loss_list = self.loss_and_update(batch)
            self.total_loss_sum += torch.sum(loss_list).item()
            self.total_loss_agents += loss_list

        # execute in environment
        states, rewards, attention_maps = self.play_step(self.epsilon)
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        if epoch % (self.max_epochs // 10) == 0 and step == (
            self.max_episode_length // 2
        ):
            for agent_id, agent in enumerate(self.agents):
                attention_map = (
                    attention_maps[agent_id]
                    .mean(dim=0)[0, :, 0, 1:]
                    .view(-1, agent.brain.patched_size_x, agent.brain.patched_size_y)
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
                        f"agent_{str(agent_id)}/attention_mean": [
                            wandb.Image(
                                data_or_path=fig,
                                caption="mean attention heatmap",
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
                            caption=f"attention heatmap from head {str(head_id)}",
                        )
                    )

                wandb.log(
                    {f"agent_{str(agent_id)}/attention_heads": fig_list},
                    step=self.global_step,
                )

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
            {
                "training_step/epsilon": self.epsilon,
                "training_step/total_reward": np.sum(rewards),
                "training_step/total_loss": self.total_loss_sum,
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
