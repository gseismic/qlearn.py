import config
import numpy as np
from qlearn.methods.iteration import (
    value_iteration_learn, 
    policy_iteration_learn,
    mc_policy_iteration_learn
)
from qlearn.envs.grid_world import GridWorld
import torch
import random
random.seed(36)
np.random.seed(36)
torch.manual_seed(36)

def test_env_gridworld():
    env = GridWorld(shape=(3, 5))
    episode, done = env.generate_episode(env.random_policy, start_state=(0,0))
    print(episode, len(episode))
    env.animate_episode(episode, interval=300)

def test_env_gridworld_episodes():
    target_state=(5,8)
    target_state=(5,8)
    def reward(state, action, next_state):
        if next_state == target_state:
            return 0
        else:
            return -1

    env = GridWorld(shape=(6, 10), target_state=target_state, fn_reward=reward)
    for i in range(5000):
        episode, done = env.generate_episode(env.random_policy, start_state=(0,0))
        if len(episode) >= 1000:
            print('...', episode[-5:], len(episode))
        # print(episode, len(episode))
        # env.animate_episode(episode, interval=300)


if __name__ == '__main__':
    if 0:
        test_env_gridworld()
    if 1:
        test_env_gridworld_episodes()
