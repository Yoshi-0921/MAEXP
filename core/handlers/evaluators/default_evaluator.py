
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
import os

from .abstract_evaluator import AbstractEvaluator

sns.set()


class DefaultEvaluator(AbstractEvaluator):
    def loop_step(self, step: int, epoch: int):
        # execute in environment
        states, rewards = self.play_step()
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        log_step = self.max_episode_length // 2
        if epoch % (self.max_epochs // 3) == 0 and step in [log_step - 1, log_step, log_step + 1]:
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

    def loop_epoch_end(self):
        self.env.accumulate_heatmap()
        self.log_scalar()
        if (self.episode_count + 1) % (self.max_epochs // 10 + 1) == 0:
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
        for i, heatmap in enumerate(heatmap_accumulated_objects):
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
                wandb.Image(data_or_path=fig, caption=f"Object {i} generated")
            )
            plt.close()

        # log heatmap_events_left FIXME only object 0 is plotted
        for i, heatmap in enumerate(self.env.heatmap_accumulated_objects_left):
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
                wandb.Image(data_or_path=fig, caption=f"Object {i} left")
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
