import pytest
import numpy as np
import gymnasium as gym
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.dqn.main import DQNAgent_Main

seed_all(42)

def test_c51_basic():
    # 创建一个简单的网格世界环境
    env = grid_world.GridWorldEnv_V2(shape=(6, 10), start_state=(0,0), goal_state=(5,8))
    
    # 设置C51算法的配置
    config = {
        'algorithm': 'c51',
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'memory_size': 1000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 10,
        'dueling_dqn': False,
        'double_dqn': True,
        'prioritized_replay': True,
    }
    
    # 创建C51代理
    agent = DQNAgent_Main(env, config)
    
    # 训练代理
    num_episodes = 200
    max_steps_per_episode = 100
    
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            agent.update()
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    # 计算平均奖励
    avg_reward = np.mean(total_rewards[-20:])  # 最后20个episode的平均奖励
    print(f"C51 算法在 {num_episodes} 个episode后的平均奖励: {avg_reward}")
    
    # 验证学习是否有效
    assert avg_reward > -50, f"C51算法性能不佳，平均奖励为 {avg_reward}"
    
    # 测试模型保存和加载
    agent.save("c51_test_model.pth")
    loaded_agent = DQNAgent_Main(env, config)
    loaded_agent.load("c51_test_model.pth")
    
    # 比较原始代理和加载的代理在相同状态下的动作选择
    test_state = env.reset()[0]
    original_action = agent.select_action(test_state)
    loaded_action = loaded_agent.select_action(test_state)
    assert original_action == loaded_action, "加载的模型与原始模型行为不一致"

if __name__ == "__main__":
    test_c51_basic()
