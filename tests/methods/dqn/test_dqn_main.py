import config
import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.dqn.main import DQNAgent_Main

seed_all(36)

# 还可以为动作空间和观察空间设置随机种子 | Set the seed for action and observation spaces
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
def test_dqn_main():
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
    }
    agent = DQNAgent_Main(env, config=config)
    # agent.set_config(config=config)
    # 退出条件： 参数控制学习的总数，最大步数，达到目标奖励
    # | Exit condition: parameters control the total number of learning, maximum steps, and target reward
    agent.learn(num_episodes=1000, max_step_per_episode=500, max_total_steps=100000, target_reward=500, seed=seed)
    # agent.learn(num_episodes=1000, max_step_per_episode=500, max_total_steps=100000, target_reward=300, seed=seed)
    
    # 保存模型 | Save the model
    # agent.save("dqn_main_agent.pth")

if __name__ == '__main__':
    if 1:
        test_dqn_main()
