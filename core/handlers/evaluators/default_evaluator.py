import numpy as np
import seaborn as sns
import wandb

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
                if self.config.agent_tasks[int(agent_id)] == "-1":
                    continue
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
        if (self.episode_count + 1) % max(1, self.max_epochs // 10) == 0:
            self.log_heatmap()
        self.reset()
