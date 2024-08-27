import config
import numpy as np
from qlearn.methods.iteration import (
    value_iteration_learn, 
    policy_iteration_learn
)
from qlearn.envs.grid_world import GridWorld
import torch
torch.manual_seed(36)

def test_grid_world():
    env = GridWorld(shape=(3, 5))
    episode = env.generate_episode(env.random_policy)
    print(episode)
    # 动画
    # env.animate_episode(episode, interval=100)

def test_value_iteration_basic():
    target_state=(5,8)
    def reward(state, action, next_state):
        if next_state == target_state:
            return 0
        else:
            return -1
    env = GridWorld(shape=(6, 10), target_state=target_state, fn_reward=reward)

    initial_state_values = torch.zeros(len(env.observation_space))
    # initial_state_values = torch.randn(len(env.observation_space))
    gamma = 0.9
    exit_code, policy_table, (Q_table, state_values) = value_iteration_learn(
        env, initial_state_values, gamma, eps_exit=1e-6,
        max_iter= 100, logger=None, verbose_freq=1
    )

    print(state_values)
    # env.plot_state_values(state_values)

    def policy(state):
        i_state = env.index_of_state(state)
        max_index = torch.argmax(policy_table[i_state, :])
        action = env.action_space[max_index.item()]
        return action

    print(f'{Q_table=}')
    print(f'{policy_table=}')
    episode = env.generate_episode(policy)
    print(f'{episode=}')
    # 动画
    # env.animate_episode(episode, interval=500)

def test_policy_iteration_basic():
    target_state=(5,8)
    def reward(state, action, next_state):
        if next_state == target_state:
            return 0
        else:
            return -1
    env = GridWorld(shape=(6, 10), target_state=target_state, fn_reward=reward)

    initial_policy_state_values = torch.zeros(len(env.observation_space))
    # initial_state_values = torch.randn(len(env.observation_space))
    initial_policy_table = torch.randn((len(env.observation_space),
                                  len(env.action_space)))
    # print(f'before: {initial_policy_table=}')
    initial_policy_table = torch.softmax(initial_policy_table, dim=1)
    # print(f'after: {initial_policy_table=}')
    gamma = 0.01 # 无法产生正确的结果
    gamma = 0.1 # 数值越小，越向目标space集中
    gamma = 0.5 # 数值越小，越向目标space集中，收敛越快
    verbose = 1
    exit_code, policy_table, (Q_table, state_values) = policy_iteration_learn(
        env, initial_policy_table, gamma, 
        eps_exit=1e-6,
        j_eps_exit=1e-3,
        j_trunc = 100,
        max_iter= 100, logger=None, verbose_freq=1,
        verbose=verbose
    )

    print(f'{state_values=}')
    env.plot_state_values(state_values)

    def policy(state):
        i_state = env.index_of_state(state)
        max_index = torch.argmax(policy_table[i_state, :])
        action = env.action_space[max_index.item()]
        return action

    # print(f'{Q_table=}')
    # print(f'{policy_table=}')
    episode = env.generate_episode(policy)
    print(f'{episode=}')
    # 动画
    env.animate_episode(episode, interval=500)


if __name__ == '__main__':
    if 0:
        test_grid_world()
    if 0:
        test_value_iteration_basic()
    if 1:
        test_policy_iteration_basic()
