import gymnasium as gym
import matplotlib.pyplot as plt
from qac_agent import QACAgent
from evals import eval_agent_performance
import os
from pathlib import Path

def main():
    env = gym.make("CartPole-v1")
    
    # 使用混合类型配置
    config = {
        'model_type': 'IndependentActorCriticMLP',
        'model_kwargs': {
            'hidden_sizes': [64, 64],
            'activation': 'relu',
        },
        'optimizer': 'adam',
        'optimizer_kwargs': {'lr': 0.0003},
        'gamma': 0.99  # 明确指定gamma值
    }
    
    agent = QACAgent(env, config=config)

    # 创建学习参数字典
    learn_params = {
        'max_episodes': 100,
        'max_total_steps': 2000,
        'max_episode_steps': None,
        'max_runtime': None,
        'reward_threshold': None,
        'reward_window_size': 100,
        'min_reward_threshold': None,
        'max_reward_threshold': None,
        'reward_check_freq': 10,
        'verbose_freq': 10,
        'no_improvement_threshold': 50,
        'improvement_threshold': None,
        'improvement_ratio_threshold': None,
        'checkpoint_freq': 100,
        'checkpoint_path': Path('checkpoints'),
        'final_model_path': Path('models') / 'final_model.pth',
        'lang': 'zh'
    }

    training_info = agent.learn(**learn_params)

    print(f"Final model saved at: {training_info['final_model_path']}")

    # 绘制训练过程中的奖励
    plt.figure(figsize=(10, 5))
    plt.plot(training_info['rewards_history'])
    plt.title('Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('training_rewards.png')
    plt.show()

    # 使用新的测试函数
    lang = 'zh'  # 或 'en'，根据需要选择语言
    performance_stats = eval_agent_performance(agent, env, num_episodes=10, lang=lang)
    for key, value in performance_stats.items():
        print(f"{key}: {value}")

    env.close()

if __name__ == "__main__":
    main()