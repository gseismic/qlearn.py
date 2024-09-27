import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from .network import PolicyNetwork
from rlearn.methods.mcpg.base_agent import BaseAgent
from rlearn.methods.utils.monitor import RewardMonitor
from rlearn.utils.seed import seed_all

class MCPGAgent_Naive(BaseAgent):
    schema = [
        dict(field='learning_rate', required=False, default=0.01, rules=dict(type='float', gt=0)),
        dict(field='gamma', required=False, default=0.99, rule=dict(type='float', min=0, max=1)),
        dict(field='normalize_return', required=False, default=True, rules=dict(type='bool')),
        dict(field='eps', required=False, default=1e-8, rules=dict(type='float', gt=0)),
        dict(field='verbose_freq', required=False, default=10, rules=dict(type='int', gt=0)),
        dict(field='policy.hidden_sizes', required=False, default=[64, 64], rules=dict(type='list', min_len=1)),
        dict(field='policy.activation', required=False, default='tanh', rules=dict(type='str', choices=['tanh', 'relu'])),
        dict(field='policy.init_method', required=False, default='kaiming', rules=dict(type='str', choices=['uniform', 'xavier', 'kaiming'])),
    ]
    
    def __init__(self, env, config=None, logger=None, seed=None):
        super().__init__(env, config, logger)
        self.input_dim = np.prod(env.observation_space.shape)
        self.output_dim = env.action_space.n
        if seed is not None:
            seed_all(seed)
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
        # 这里涉及随机性
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])
        self.gamma = self.config['gamma']
        self.eps = self.config['eps']
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def calculate_returns(self, rewards):
        returns = []
        R = 0
        # rewards: list[float] length: T
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns = torch.tensor(returns[::-1])
        if self.config['normalize_return']:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns
    
    def update_policy(self, states, actions, returns: torch.Tensor):
        self.optimizer.zero_grad()  
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        
        probs = self.policy(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        loss = -(log_probs * returns).mean()
        
        loss.backward()
        self.optimizer.step()
    
    def learn(self, 
              num_episodes, 
              max_step_per_episode=None, 
              max_total_steps=None, 
              target_episode_reward=None, 
              target_window_avg_reward=None, 
              target_window_length=None,
              seed=None):
        if seed is not None:
            seed_all(seed)
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
            
        self.monitor = RewardMonitor(
            max_step_per_episode=max_step_per_episode,
            max_total_steps=max_total_steps,
            target_episode_reward=target_episode_reward,
            target_window_avg_reward=target_window_avg_reward,
            target_window_length=target_window_length
        )
        should_stop = False
        for episode_idx in range(num_episodes):
            state, _ = self.env.reset()
            self.monitor.before_episode_start()

            states, actions, rewards = [], [], []
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.monitor.after_step_env(next_state, reward, terminated, truncated, info)
                
                done = terminated or truncated
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                exit_episode, exit_learning, (exit_learning_code, exit_learning_msg) = self.monitor.check_exit_conditions()

                if exit_learning:
                    if exit_learning_code == 0:
                        should_stop = True
                        break
                    elif exit_learning_code >= 1:
                        should_stop = True
                        break
                    else:
                        raise ValueError(f"Invalid exit learning code: {exit_learning_code}")
                
                if exit_episode:
                    break
            
            self.monitor.after_episode_end()
            returns = self.calculate_returns(rewards)
            self.update_policy(states, actions, returns)
                
            if (episode_idx + 1) % self.config['verbose_freq'] == 0:
                self.logger.info(f"Episode {episode_idx+1}/{num_episodes}, Episode Reward: {self.monitor.episode_reward}")
   
            if should_stop:
                if exit_learning_code == 0:
                    self.logger.info(exit_learning_msg)
                elif exit_learning_code >= 1:
                    self.logger.warning(exit_learning_msg)
                else:
                    raise ValueError(f"Invalid exit learning code: {exit_learning_code}")
                break
            
            if episode_idx == num_episodes - 1:
                self.logger.warning(f"Reached the maximum number of episodes: {num_episodes}")
        
        exit_info = {
            "reward_list": self.monitor.all_episode_rewards
        }
        return exit_info