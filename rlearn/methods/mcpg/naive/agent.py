import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from .network import PolicyNetwork

class MCPGAgent_Naive:
    DEFAULT_CONFIG = {
        'learning_rate': 0.01,
        'gamma': 0.99,
    }
    def __init__(self, env, config):
        self.env = env
        self.input_dim = np.prod(env.observation_space.shape)
        self.output_dim = env.action_space.n
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])
        self.gamma = self.config['gamma']
    
    def select_action(self, state):
        # Select an action based on the policy / 根据策略选择动作
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def calculate_returns(self, rewards):
        # Calculate discounted returns / 计算折扣回报
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update_policy(self, states, actions, returns):
        # Update the policy using the REINFORCE algorithm / 使用REINFORCE算法更新策略
        self.optimizer.zero_grad()
        # print(f'{len(states)=}, {states[0].shape=}')
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        
        probs = self.policy(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        loss = -(log_probs * returns).mean()
        
        loss.backward()
        self.optimizer.step()
    
    def learn(self, num_episodes, seed=None):
        # Train the agent for a specified number of episodes / 训练智能体指定数量的回合
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            states, actions, rewards = [], [], []
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
            
            returns = self.calculate_returns(rewards)
            self.update_policy(states, actions, returns)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Total Reward: {sum(rewards)}")
