import torch
torch.autograd.set_detect_anomaly(True)
import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.dqn.main import DQNAgent_Main
import pytest
import numpy as np

seed_all(42)

def test_c51_basic():
    env = gym.make('CartPole-v1')
    seed = 42
    
    config = {
        'algorithm': 'c51',
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'memory_size': 10000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 10,
        'dueling_dqn': False,
        'double_dqn': True,
        'prioritized_replay': True,
        'verbose_freq': 1,
        'use_grad_clip': True,
        'max_grad_norm': 10,
    }
    
    agent = DQNAgent_Main(env, config=config)
    exit_info = agent.learn(num_episodes=200, 
                max_step_per_episode=500, 
                max_total_steps=100000, 
                target_episode_reward=195,
                seed=seed)
    
    assert 'reward_list' in exit_info
    assert len(exit_info['reward_list']) > 0
    
    # 计算最后20个episode的平均奖励
    avg_reward = np.mean(exit_info['reward_list'][-20:])
    print(f"C51 算法在训练后的平均奖励: {avg_reward}")
        
    # 测试模型保存和加载
    agent.save("c51_test_model.pth")
    loaded_agent = DQNAgent_Main(env, config)
    loaded_agent.load("c51_test_model.pth")
    
    # 比较原始代理和加载的代理在相同状态下的动作选择
    test_state, _ = env.reset(seed=seed)
    original_action = agent.select_action(test_state)
    loaded_action = loaded_agent.select_action(test_state)
    assert original_action == loaded_action, "加载的模型与原始模型行为不一致"

def test_c51_noisy_net():
    env = gym.make('CartPole-v1')
    seed = 42
    
    # 对于CartPole环境，考虑到其特性（每步奖励为1，episode最长500步），一个合理的选择可能是：
    # v_max = 200  # 略高于最大可能累积奖励（500）
    config = {
        'algorithm': 'c51',
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 100, # 
        'learning_rate': 0.001,
        'batch_size': 32,
        'memory_size': 10000,
        'gamma': 0.99,
        'target_update_freq': 10,
        'dueling_dqn': False,
        'double_dqn': True,
        'prioritized_replay': True,
        'verbose_freq': 1,
        'use_grad_clip': True,
        'max_grad_norm': 10,
        'use_noisy_net': True,
        'noisy_net_type': 'factorized',
        'noisy_net_std_init': 0.5,
        'noisy_net_k': 2,
        'noise_decay': 0.99,
        'min_exploration_factor': 0.1,
    }
    
    agent = DQNAgent_Main(env, config=config)
    exit_info = agent.learn(num_episodes=300, 
                max_step_per_episode=500, 
                max_total_steps=150000, 
                target_episode_reward=195,
                seed=seed)
    
    assert 'reward_list' in exit_info
    assert len(exit_info['reward_list']) > 0
    
    avg_reward = np.mean(exit_info['reward_list'][-20:])
    print(f"C51 with Noisy Net 算法在训练后的平均奖励: {avg_reward}")
    
if __name__ == "__main__":
    if 0:
        test_c51_basic()
    if 1:
        test_c51_noisy_net()