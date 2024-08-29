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

def make_gridworld0():
    target_state=(1,2)
    def reward(state, action, next_state):
        return 0 if next_state == target_state else -1
    env = GridWorld(shape=(2, 3), target_state=target_state, fn_reward=reward)
    return env

def make_gridworld1():
    target_state=(5,8)
    def reward(state, action, next_state):
        return 0 if next_state == target_state else -1
    env = GridWorld(shape=(6, 10), target_state=target_state, fn_reward=reward)
    return env


def test_mc_raw():
    pass


def test_mc_policy_iteration_basic():
    env = make_gridworld0()
    initial_policy_table = torch.randn((len(env.observation_space),
                                  len(env.action_space)))
    # print(f'before: {initial_policy_table=}')
    initial_policy_table = torch.softmax(initial_policy_table, dim=1)
    # print(f'after: {initial_policy_table=}')
    gamma = 0.9 # 数值越小，越向目标space集中，收敛越快
    initial_Q_table = 'zeros'
    initial_Q_table = 'randn'
    eps_explore = 0.1
    num_episodes = 20000
    num_episodes = 10000
    num_episodes = 5000
    num_episodes = 500
    exit_code, policy_table, (Q_table, state_values), info = mc_policy_iteration_learn(
        env, initial_policy_table,
        gamma=gamma,
        num_episodes=num_episodes,
        initial_Q_table=initial_Q_table,
        episode_start_state=None,
        eps_explore=eps_explore,
        logger=None,
        verbose_freq=1,
        verbose=1
    )

    print(env.shape)
    visited = info['visited']
    print(f'{dict(visited)=}')
    print(f'{sorted(visited.keys())=}')

    def policy(state):
        i_state = env.index_of_state(state)
        max_index = torch.argmax(policy_table[i_state, :])
        # print(f'{policy_table[i_state, :]=}')
        # print(f'{max_index=}')
        action = env.action_space[max_index.item()]
        return action

    for i_state in range(Q_table.shape[0]):
        state = env.state_space[i_state]
        for i_action in range(Q_table.shape[1]):
            action = env.action_space[i_action]
            print(f'{state, action=}, {Q_table[i_state, i_action]=}')

    # print(f'{Q_table=}')
    print(f'{policy_table=}')
    episode, done = env.generate_episode(policy, start_state=(0,0), max_size=20)
    print(f'{episode=}')
    # 动画
    env.animate_episode(episode, interval=500)

if __name__ == '__main__':
    if 1:
        test_mc_policy_iteration_basic()
