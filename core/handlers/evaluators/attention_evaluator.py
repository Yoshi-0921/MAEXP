import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb

from .default_evaluator import DefaultEvaluator

sns.set()


class AttentionEvaluator(DefaultEvaluator):
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

        rewards, _, new_states = self.env.step(actions, self.order)

        self.states = new_states

        return states, rewards, attention_maps

    def loop_step(self, step: int, epoch: int):
        # execute in environment
        states, rewards, attention_maps = self.play_step()
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        log_step = self.max_episode_length // 2
        if epoch % (self.max_epochs // 3) == 0 and step in [
            log_step - 1,
            log_step,
            log_step + 1,
        ]:
            for agent_id, agent in enumerate(self.agents):
                for attention_map, image, view_method in zip(
                    attention_maps[agent_id],
                    self.env.observation_handler.render(states[agent_id]),
                    ["local", "relative"],
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
                        annot_kws={"fontsize": 8},
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
                            annot_kws={"fontsize": 8},
                        )
                        fig_list.append(
                            wandb.Image(
                                data_or_path=fig,
                                caption=f"{view_method} attention heatmap from head {str(head_id)}",
                            )
                        )

                    wandb.log(
                        {f"agent_{str(agent_id)}/{view_method}_attention_heads": fig_list},
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
            {"training_step/total_reward": np.sum(rewards)},
            step=self.global_step,
        )

        for agent_id, reward in enumerate(rewards):
            wandb.log(
                {f"agent_{str(agent_id)}/step_reward": reward},
                step=self.global_step,
            )
