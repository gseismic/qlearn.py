import torch
torch.autograd.set_detect_anomaly(True)
import gymnasium as gym
import pytest
from rlearn.utils.seed import seed_all
from rlearn.methods.dqn.main import DQNAgent_Main


seed_all(36)

def test_dqn_main_default():
    env = gym.make('CartPole-v1')
    seed = 36
    config = {
        'prioritized_replay': True,
        'double_dqn': True,
        'dueling_dqn': True,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.995,
        'gamma': 0.99,
        'verbose_freq': 1,
        'use_grad_clip': True,
        'max_grad_norm': 10,
    }
    agent = DQNAgent_Main(env, config=config)
    exit_info = agent.learn(num_episodes=100, 
                max_step_per_episode=500, 
                max_total_steps=10000, 
                target_episode_reward=200, 
                seed=seed)
    
    assert 'reward_list' in exit_info
    assert len(exit_info['reward_list']) > 0
    assert max(exit_info['reward_list']) >= 200

def test_dqn_main_noisy_net_dense():
    env = gym.make('CartPole-v1')
    seed = 36
    config = {
        'learning_rate': 0.001,
        'prioritized_replay': True,
        'double_dqn': True,
        'dueling_dqn': True,
        'gamma': 0.99,
        'verbose_freq': 1,
        'use_grad_clip': True,
        'max_grad_norm': 10,
        'use_noisy_net': True,
        'noisy_net_type': 'dense',
        # 'noisy_net_std_init': 0.5,
        'noisy_net_std_init': 1.0, # 太小导致无法收敛
    }
    # 全量噪音收敛比较慢
    agent = DQNAgent_Main(env, config=config)
    exit_info = agent.learn(num_episodes=500, 
                max_step_per_episode=2000,
                max_total_steps=10000, 
                target_episode_reward=200, 
                seed=seed)
    
    assert 'reward_list' in exit_info
    assert len(exit_info['reward_list']) > 0
    assert max(exit_info['reward_list']) >= 200

def test_dqn_main_noisy_net_factorized():
    env = gym.make('CartPole-v1')
    seed = 36
    config = {
        'prioritized_replay': True,
        'double_dqn': True,
        'dueling_dqn': True,
        'gamma': 0.99,
        'verbose_freq': 1,
        'use_grad_clip': True,
        'max_grad_norm': 10,
        'use_noisy_net': True,
        'noisy_net_type': 'factorized',
        'noisy_net_std_init': 0.5,
        'noisy_net_k': 2,
    }
    agent = DQNAgent_Main(env, config=config)
    exit_info = agent.learn(num_episodes=100, 
                max_step_per_episode=500, 
                max_total_steps=10000, 
                target_episode_reward=200, 
                seed=seed)
    
    assert 'reward_list' in exit_info
    assert len(exit_info['reward_list']) > 0
    assert max(exit_info['reward_list']) >= 200

def test_dqn_main_invalid_config():
    env = gym.make('CartPole-v1')
    config = {
        'use_noisy_net': True,
        'noisy_net_type': 'invalid_type',
    }
    with pytest.raises(ValueError):
        DQNAgent_Main(env, config=config)

if __name__ == '__main__':
    if 0:
        # OK
        test_dqn_main_default()
    if 0:
        # OK
        test_dqn_main_noisy_net_factorized()   
    if 1:
        test_dqn_main_noisy_net_dense()
    if 0:
        test_dqn_main_invalid_config()
    # pytest.main([__file__])