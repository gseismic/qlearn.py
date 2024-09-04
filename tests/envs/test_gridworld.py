import config
import numpy as np
from rlearn.envs import grid_world
import torch
import random
random.seed(36)
np.random.seed(36)
torch.manual_seed(36)

def test_env_gridworld():
    env = grid_world.GridWorldEnv(shape=(3, 5))
    policy = env.make_random_policy()
    episode, done = env.generate_episode(policy, start_state=(0,0))
    print(episode, len(episode))
    # grid_world.animate_episode(env, episode, interval=300)

def test_env_gridworld_episodes():
    env = grid_world.GridWorldEnv(shape=(3, 5))
    for i in range(5000):
        episode, done = env.generate_episode(env.make_random_policy(), start_state=(0,0))
        if len(episode) >= 1000:
            print('...', episode[-5:], len(episode))
        # print(episode, len(episode))
        # env.animate_episode(episode, interval=300)


if __name__ == '__main__':
    if 0:
        test_env_gridworld()
    if 1:
        test_env_gridworld_episodes()
