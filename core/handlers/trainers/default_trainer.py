import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

from .abstract_trainer import AbstractTrainer

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
sns.set()


class DefaultTrainer(AbstractTrainer):
    def loop_epoch_start(self, epoch: int):
        if epoch == (self.max_epochs // 2):
            self.save_state_dict(epoch=epoch)

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
        states, rewards, dones = self.play_step(self.epsilon)
        self.episode_reward_sum += np.sum(rewards)
        self.episode_reward_agents += np.asarray(rewards)

        if epoch % (self.max_epochs // 4) == 0 and step == (
            self.max_episode_length // 2
        ):
            # log attention_maps of agent0
            for agent_id in range(len(self.agents)):
                if self.config.agent_tasks[int(agent_id)] == "-1":
                    continue

                self.log_destination_channel(agent_id)

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
                "training/total_loss": self.step_loss_sum,
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

    def loop_epoch_end(self):
        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.config.epsilon_end, self.epsilon)

        if self.episode_count % self.synchronize_frequency == 0:
            for agent in self.agents:
                agent.synchronize_brain()

        self.env.accumulate_heatmap()
        self.log_scalar()
        if (self.episode_count + 1) % max(1, self.max_epochs // 10) == 0:
            self.log_heatmap()
        self.reset()
