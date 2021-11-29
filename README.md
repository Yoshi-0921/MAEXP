# MAEXP-MADRL

## 0. What is this repo for?
This repository is for experiments to investigate unknown features and potential advantages of multi-agent reinforcement learning with deep neaural network.
Code is basically written in Python using PyTorch.
<p align="center"><img width="250" alt="spread_maddpg_notag" src="https://user-images.githubusercontent.com/60799014/92319743-64f73e00-f056-11ea-9bac-cdeadc4cc2bd.gif"></p>

GIF cited from [here](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/)

## 1. Implemented so far
- DQN
- C51
- QR-DQN
- IQN
- DA3
- DA6

# 2. Config for experiments
```yaml
defaults:
  - model: {mlp, conv_mlp, da3, da3_baseline, da6}
  - map: {simple, four_rooms, four_rectangle, three_rooms}
  - opt: {rmsprop, adam}
  - job_logging: disabled

project_name: str{type the name of project}
name: str{type the name of run}
gpu: int{gpu index}
# ==================================================
# configure environment

world: 'default'
environment: {'default', 'shared_mean_reward', 'shared_max_reward', 'object_types', 'da3_types_test'}
seed: str{type seed}
num_objects: int{type the number of objects}
type_objects: int{type the number of types of objects}
objects_color: list(str){type the color of objects in a list}
# ==================================================
# configure agent

agent_type: {'default', 'attention'}
brain: {'dqn', 'da3', 'da3_baseline', 'da6'}
num_agents: int{type the number of agents}
agents_color: list(str){type the color of each agent in a list}
init_xys: list(list(int)){type the initial coordinates of each agent in a list}
shuffle_init_xys: {True, False}
phase: {'training', 'evaluation'}
# ==================================================
# configure observation

visible_range: {5, 7, 9}
transparent_observation: {True, False}
destination_channel: {True, False}
reset_destination_period: int{type epoch to reset the destination area}
agent_view_method: {'default' 'self', 'simple', 'transition', 'individual'}
object_view_method: {'default', 'simple', 'object_types'}
observation_area_mask: {'local', 'relative', 'merged'}
observation_noise: {False, 'sensing_dist', 'threshold_sensing_dist', 'flat', 'flip'}
noise_threshold_ratio: float{type the threshold of noise}
flip_noise_probabilities: list(float){type the probabilities of flip noise}
std_distribution: list(float){type the standad deviation of sensing noise}
past_step: int{type the number of time steps to observe until current state}

# ==================================================
# configure training

trainer: {'default', 'attention', 'attention_wandering'}
evaluator: {'default', 'attention', 'attention_wandering', 'da3_video', 'da3_wandering_video'}
max_epochs: int{type the number of maximum training epochs}
max_episode_length: int{type the number of maximum training steps per epoch}
batch_size: int{type batch size for training}
populate_steps: int{type number of steps to populate the replay buffer}
capacity: int{type the capacity of replay buffer}
gamma: float{type the discout factor of reward}
synchronize_frequency: int{type the frequency to synchronize the network and its target network}
epsilon_decay: float{type the ratio to decay epsilon}
epsilon_initial: float{type the initial value of epsilon}
epsilon_end: float{type the terminal value of epsilon}
# ==================================================
# configre evaluation

validate_epochs: int{type the number of maximum evaluation epochs}
pretrained_weight_path: str{type the path to pretrained weight via wandb API, otherwise False}
```
