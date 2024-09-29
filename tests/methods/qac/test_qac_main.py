import torch
torch.autograd.set_detect_anomaly(True)
import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.qac.main import QACAgent
import pytest
import numpy as np
from rlearn.methods.utils.monitor import get_monitor

def test_qac_main():
    env = gym.make('CartPole-v1')
    config = {
        'policy_net': {
            'type': 'MLP',
               'params': {
                'hidden_dims': [128, 128],
                'activation': 'relu',
                'init_type': 'kaiming_uniform',
                'use_noisy': False,
                'factorized': True,
                'rank': 1,
                'std_init': 0.4,
                'use_softmax': True,
            },
        },
        'optimizer': {
            'type': 'Adam',
            'params': {
                'lr': 0.01,
            }
        },
        'gamma': 0.99,
        'monitor': {
            'type': 'reward',
            'params': {
                'max_step_per_episode': 600,
                'max_total_steps': 10000,
                'target_episode_reward': 195,
                'target_window_avg_reward': 195,
                'target_window_length': 100,
                # 'target_window_type': 'moving_average',
            }
        },
        'verbose_freq': 1
    }
    seed = 36
    agent = QACAgent(env, config=config, seed=seed)
    # monitor = get_monitor(config['monitor'])
    # agent.set_monitor(monitor)
    exit_info = agent.learn(num_episodes=500)
    print(f'{exit_info=}')
    rewards = evaluate_agent(agent, env)
    print(f'{rewards=}')
    assert rewards > 195
    
def evaluate_agent(agent, env, n_episodes=10):
    total_rewards = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward
    return total_rewards / n_episodes

if __name__ == '__main__':
    if 1:
        test_qac_main()