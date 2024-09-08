import config
import torch
import gymnasium as gym
from rlearn.envs import grid_world
from rlearn.utils.seed import seed_all
from rlearn.methods.dqn.c51 import DQNAgent_C51
from rlearn.nets import MLP
import numpy as np
seed_all(36)

def test_c51_basic():    
    env = grid_world.GridWorldEnv_V2(shape=(6, 10), start_state=(0,0), goal_state=(5,8))
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    
    model = MLP(input_size=state_dim, hidden_sizes=[128, 128], output_size=action_dim)
    agent = DQNAgent_C51(name='test', env=env, model=model)
    rewards = agent.learn(learning_rate=0.001, max_epochs=1000) # total_timesteps=10000)
    
    print(f"平均奖励: {np.mean(rewards)} | Average reward: {np.mean(rewards)}")
    
    # 保存模型 | Save the model
    agent.save("q_agent.pth")

if __name__ == '__main__':
    if 1:
        test_c51_basic()
