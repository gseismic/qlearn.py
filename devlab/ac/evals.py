import numpy as np
import time
from i18n import get_text


def eval_agent_performance(agent, 
                           env, 
                           num_episodes=100, 
                           max_steps=1000, 
                           deterministic=False,
                           lang='en'):
    """
    测试agent的性能
    
    参数:
    - agent: 要测试的智能体
    - env: 环境
    - num_episodes: 测试的回合数
    - max_steps: 每个回合的最大步数
    
    返回:
    - 包含性能统计信息的字典
    """
    total_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action, _ = agent.predict(state, deterministic=deterministic)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    # 计算统计信息
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    max_reward = np.max(total_rewards)
    min_reward = np.min(total_rewards)
    avg_episode_length = np.mean(episode_lengths)
    
    # 计算成功率（假设奖励大于某个阈值为成功）
    success_threshold = env.spec.reward_threshold if hasattr(env.spec, 'reward_threshold') else avg_reward
    success_rate = sum(r >= success_threshold for r in total_rewards) / num_episodes
    
    return {
        get_text('average_reward', lang): avg_reward,
        get_text('reward_std', lang): std_reward,
        get_text('max_reward', lang): max_reward,
        get_text('min_reward', lang): min_reward,
        get_text('average_episode_length', lang): avg_episode_length,
        get_text('success_rate', lang): success_rate,
        get_text('test_episodes', lang): num_episodes,
        get_text('test_duration', lang): test_duration
    }
