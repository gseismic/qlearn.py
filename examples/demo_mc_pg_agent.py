import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.mcpg.naive import MCPGAgent_Naive

seed = 36
seed_all(seed)
env = gym.make('CartPole-v1')

config = {
    'learning_rate': 0.01,
    'gamma': 0.99
}
agent = MCPGAgent_Naive(env, config=config)
# TODO
# agent.learn(num_episodes=1000, max_step_per_episode=500, max_total_steps=100000, target_reward=500, seed=seed)
agent.learn(num_episodes=1000, seed=seed)