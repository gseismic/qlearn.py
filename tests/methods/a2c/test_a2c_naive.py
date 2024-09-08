import config
import gymnasium as gym
from rlearn.utils.seed import seed_all
from rlearn.methods.a2c.naive import A2CAgent_Naive

seed_all(36)

# 还可以为动作空间和观察空间设置随机种子 | Set the seed for action and observation spaces
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
def test_a2c_basic():
    env = gym.make('CartPole-v1')
    seed = 36
    config = {
        'learning_rate': 0.001,
        'gamma': 0.99
    }
    agent = A2CAgent_Naive(env, config=config)
    # TODO
    # agent.learn(num_episodes=1000, max_step_per_episode=500, max_total_steps=100000, target_reward=500, seed=seed)
    agent.learn(num_episodes=1000, seed=seed)

if __name__ == '__main__':
    if 1:
        test_a2c_basic()
