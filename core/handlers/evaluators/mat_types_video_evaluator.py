import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import io
from omegaconf import DictConfig

from .default_evaluator import DefaultEvaluator

plt.rcParams["figure.facecolor"] = 'gray'
plt.rcParams['savefig.facecolor'] = 'gray'
# sns.set()


class MATTypesVideoEvaluator(DefaultEvaluator):
    def __init__(self, config: DictConfig, environment):
        super().__init__(config, environment)

        width = 640
        hieght = 480
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter('output.mp4', fourcc, float(fps), (width, hieght))

        self.images = []

    def endup(self):
        self.video.release()

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        actions = [[] for _ in range(self.env.num_agents)]
        attention_maps = [[] for _ in range(self.env.num_agents)]
        random.shuffle(self.order)

        states = self.states
        for agent_id in self.order:
            if agent_id in [4, 5]:
                actions[agent_id] = self.agents[agent_id].get_random_action()
                continue

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

        image = np.stack((self.env.world.map.wall_matrix.astype(np.float),) * 3, axis=-1)
        image[..., 0] += self.env.world.map.objects_matrix.astype(np.float)
        image[..., 1] += self.env.world.map.objects_matrix.astype(np.float)
        for agent_id, agent in enumerate(self.env.agents):
            pos_x, pos_y = self.env.world.map.coord2ind(agent.xy)
            if agent_id in [4, 5]:
                image[pos_x, pos_y, 0] = 1
            else:
                image[pos_x, pos_y, 2] = 1
        image *= 255.0
        image = image.astype(np.int)
        self.images.append(image.transpose((1, 0, 2)))
        if step == 0:
            self.images.append(image.transpose((1, 0, 2)))

        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(self.images.pop(0))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title(f'Step {step}')

        agent_id, agent = 0, self.agents[0]
        for agent_id, agent in enumerate(self.agents):
            if agent_id in [4, 5]:
                break

            plt.subplot(4, 4, agent_id + 1 + 8)
            image = self.env.observation_handler.render(states[agent_id]).numpy().transpose((1, 2, 0))
            image *= 255.0
            image = image.astype(np.int)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.title(f'Agent {agent_id}')

            plt.subplot(4, 4, agent_id + 5 + 8)
            attention_map = (
                attention_maps[agent_id]
                .mean(dim=0)[0, :, 0, 1:]
                .view(-1, agent.brain.patched_size_x, agent.brain.patched_size_y)
                .cpu()
            )
            if agent_id == 3:
                sns.heatmap(
                    torch.t(attention_map.mean(dim=0)),
                    cmap='bone_r',
                    vmin=0,
                    square=True,
                    vmax=0.25,
                )
            else:
                sns.heatmap(
                    torch.t(attention_map.mean(dim=0)),
                    cmap='bone_r',
                    square=True,
                    cbar=False,
                )
            plt.xticks([])
            plt.yticks([])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        dst2 = cv2.imdecode(enc, 1)

        self.video.write(dst2)

    def loop_epoch_end(self):
        self.env.accumulate_heatmap()
        self.log_scalar()
        self.log_heatmap()
        self.reset()
