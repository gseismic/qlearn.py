import config
import torch
import gymnasium as gym
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.dqn import DQNAgent_Naive
from rlearn.nets import MLP

seed_all(36)

def test_dqn_naive():    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = MLP(input_size=state_dim, hidden_sizes=[128, 128], output_size=action_dim)
    agent = DQNAgent_Naive(name='test', env=env, model=model)
    rewards = agent.learn(learning_rate=0.001, max_epochs=1000) # total_timesteps=10000)
    
    print(f"平均奖励: {np.mean(rewards)} | Average reward: {np.mean(rewards)}")
    
    # 保存模型 | Save the model
    agent.save("q_agent.pth")

if __name__ == '__main__':
    if 1:
        test_dqn_naive()
