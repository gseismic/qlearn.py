import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.dqn.main import DQNAgent_Main

seed = 36
seed_all(seed)

env = gym.make('CartPole-v1')
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
exit_code, info = agent.learn(num_episodes=1000, 
                                         max_step_per_episode=500, 
                                         max_total_steps=100000,
                                         target_reward=300, 
                                         seed=seed)
print(f'{exit_code=}, {info=}')
# 保存模型 | Save the model
# agent.save("dqn_main_agent.pth")