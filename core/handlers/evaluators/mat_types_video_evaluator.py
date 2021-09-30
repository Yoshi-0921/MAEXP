import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from omegaconf import DictConfig

from .default_evaluator import DefaultEvaluator

# sns.set()


class MATTypesVideoEvaluator(DefaultEvaluator):
    def __init__(self, config: DictConfig, environment):
        self.max_epochs = config.validate_epochs

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states.clone()
        for agent_id in self.order:
            if agent_id in [4, 5]:
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue

            action, attns = self.agents[agent_id].get_action_attns(
                states[agent_id], epsilon
            )
            actions[agent_id] = action
            attention_maps[agent_id] = attns

        rewards, _, new_states = self.env.step(actions)

        self.states = new_states

        return states, rewards, attention_maps

    def loop_step(self, step: int, epoch: int):
        # execute in environment
        states, rewards, attention_maps = self.play_step()
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        image = np.stack((self.env.world.map.wall_matrix.astype(np.float)*0.5,) * 3, axis=-1)
        image[..., 0] += self.env.world.map.objects_matrix.astype(np.float)
        image[..., 1] += self.env.world.map.objects_matrix.astype(np.float)
        image[..., 2] += self.env.world.map.agents_matrix.astype(np.float)
        image *= 255.0
        image = image.astype(np.int)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title(f'step {step}')
        plt.savefig('tmp.png')

        if 1:
            for agent_id, agent in enumerate(self.agents):
                if agent_id in [4, 5]:
                    continue

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
                    annot_kws={"fontsize": 8},
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
                        annot_kws={"fontsize": 8},
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
            {"training_step/total_reward": np.sum(rewards)},
            step=self.global_step,
        )

        for agent_id, reward in enumerate(rewards):
            wandb.log(
                {f"agent_{str(agent_id)}/step_reward": reward},
                step=self.global_step,
            )
