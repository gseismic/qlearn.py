import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from rlearn.methods.utils.replay_buffer import (
    Experience,
    RandomReplayBuffer, PrioritizedReplayBuffer
)
from rlearn.logger import user_logger
from rlearn.methods.dqn.main.network import DQN, DuelingDQN
from .base_agent import BaseDQNAgent    
from rlearn.methods.utils.monitor import RewardMonitor

class DQNAgent_Main(BaseDQNAgent):
    """Online DQN Agent
    DQNAgent_Main is a class for training and evaluating a DQN agent.
    
    Notes:
        - Discrete action space Only | 仅支持离散动作空间
    
    reference:
        - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        - https://github.com/Curt-Park/rainbow-is-all-you-need/tree/master?tab=readme-ov-file 
        - https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/dqn.py
    """
    
    schema = [
        dict(field='lr', required=False, default=1e-3, rules=dict(type='float', gt=0)),
        dict(field='batch_size', required=False, default=64, rules=dict(type='int', gt=0)),
        dict(field='gamma', required=False, default=0.99, rules=dict(type='float', min=0, max=1)),
        dict(field='epsilon_start', required=False, default=1.0, rules=dict(type='float', gt=0)),
        dict(field='epsilon_end', required=False, default=0.01, rules=dict(type='float', gt=0)),
        dict(field='epsilon_decay', required=False, default=0.995, rules=dict(type='float', gt=0, max=1)),
        dict(field='target_update_freq', required=False, default=10, rules=dict(type='int', gt=0)),
        dict(field='memory_size', required=False, default=10000, rules=dict(type='int', gt=0)),
        dict(field='dueling_dqn', required=False, default=True, rules=dict(type='bool')),
        dict(field='double_dqn', required=False, default=True, rules=dict(type='bool')),
        dict(field='prioritized_replay', required=False, default=True, rules=dict(type='bool')),
        dict(field='hidden_layers', required=False, default=[128, 128], rules=dict(type='list', min_len=1)),
        dict(field='device', required=False, default='cpu', rules=dict(type='str', enum=['cpu', 'cuda'])),
        dict(field='verbose_freq', required=False, default=10, rules=dict(type='int', gt=0)),
    ]
    
    def __init__(self, env, config=None, logger=None):
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.n
        super().__init__(env, config, logger)
        self.epsilon = self.config['epsilon_start']
        self.update_steps = 0
        self.logger.info(f"DQNAgent_Main initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        
    def init_networks(self):
        if self.config['dueling_dqn']:
            self.q_network = DuelingDQN(self.state_dim, self.action_dim)
            self.target_network = DuelingDQN(self.state_dim, self.action_dim)
            self.logger.info(f"DuelingDQN initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        else:
            self.q_network = DQN(self.state_dim, self.action_dim)
            self.target_network = DQN(self.state_dim, self.action_dim)
            self.logger.info(f"DQN initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['lr'])
        self.logger.info(f"Optimizer initialized with lr: {self.config['lr']}")
        
        if self.config['prioritized_replay']:   
            self.memory = PrioritizedReplayBuffer(self.config['memory_size'])
            self.logger.info(f"PrioritizedReplayBuffer initialized with memory_size: {self.config['memory_size']}")
        else:
            self.memory = RandomReplayBuffer(self.config['memory_size'])
            self.logger.info(f"RandomReplayBuffer initialized with memory_size: {self.config['memory_size']}")
        
    def select_action(self, state):
        """
        基于epsilon-greedy策略选择动作 | Select action based on epsilon-greedy policy
        Args:
            - state: 状态 | State
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: (1, state_dim)
            q_values = self.q_network(state)  # shape: (1, action_dim)
            return q_values.argmax().item()  # 返回最大Q值对应的动作 | Return the action with the maximum Q value   
    
    def update(self):
        # Update the Q-network | 更新Q网络
        if len(self.memory) < self.config['batch_size']:
            return
        
        if self.config['prioritized_replay']:
            experiences, indices, weights = self.memory.sample(self.config['batch_size'])
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.memory.sample(self.config['batch_size'])
            weights = None
        
        batch = Experience(*zip(*experiences)) # shape: (batch_size, )
        
        # state_batch = torch.FloatTensor(batch.state).to(self.device)  # shape: (batch_size, state_dim)
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)  # shape: (batch_size, state_dim)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)  # shape: (batch_size, 1)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)  # shape: (batch_size, 1)
        # next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)  # shape: (batch_size, state_dim)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)  # shape: (batch_size, state_dim)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)  # shape: (batch_size, 1)
        
        # state_batch: shape: (batch_size, state_dim)
        # action_batch: shape: (batch_size, 1)
        # q_values: shape: (batch_size, 1)
        q_values = self.q_network(state_batch).gather(dim=1, index=action_batch)  # shape: (batch_size, 1)
        
        if self.config['double_dqn']:
            # 一个网络选择动作，另一个网络计算Q值 | One network chooses action, the other network calculates Q value
            # self.q_network: (batch_size, action_dim)
            # self.q_network(next_state_batch).max(dim=1): 在维度1上找到最大值，并返回最大值和最大值的索引
            # self.q_network(next_state_batch).max(dim=1)[1]: 返回最大值的索引
            # unsqueeze(1): 在维度1上增加一个维度，使其与action_batch的维度相同
            # 选择下一个状态下的最大Q值 | Select the maximum Q value for the next state
            # next_actions = self.q_network(next_state_batch).max(dim=1)[1].unsqueeze(1)  # shape: (batch_size, 1)
            next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)  # shape: (batch_size, 1)
            next_q_values = self.target_network(next_state_batch).gather(dim=1, index=next_actions)  # shape: (batch_size, 1)
        else:
            # 只有一个网络计算Q值 | Only one network calculates Q value
            next_q_values = self.target_network(next_state_batch).max(dim=1)[0].unsqueeze(1)  # shape: (batch_size, 1)
        
        # reward_batch: shape: (batch_size, 1)
        # done_batch: shape: (batch_size, 1)
        # next_q_values: shape: (batch_size, 1)
        # 每个采样点一个值 | One value for each sample point
        # mask: 1 - done_batch: shape: (batch_size, 1)
        expected_q_values = reward_batch + (1 - done_batch) * self.config['gamma'] * next_q_values  # shape: (batch_size, 1)
        
        # # 计算逐项损失 | Compute element-wise loss
        loss = nn.MSELoss(reduction='none')(q_values, expected_q_values.detach())
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.config['prioritized_replay']:
            td_errors = (q_values - expected_q_values).abs().detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)
        
        self.update_steps += 1
        if self.update_steps % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.config['epsilon_end'], self.epsilon * self.config['epsilon_decay'])
    
    def learn(self, 
              num_episodes, 
              max_step_per_episode=None, 
              max_total_steps=None, 
              target_episode_reward=None, 
              target_window_avg_reward=None,
              target_window_length=None,
              seed=None):
        """
        Args: 
            - num_episodes: 训练的次数 | Number of episodes to train
            - max_step_per_episode: 每个episode的最大步数 | Maximum steps per episode
            - max_total_steps: 总的训练步数 | Total steps to train
            - target_reward: 目标奖励 | Target reward to achieve
            - seed: 随机种子 | Random seed
        Returns: None
        Notes: 
            - 当设置max_total_steps时，num_episodes和max_step_per_episode将失效
            - 当设置target_reward时，num_episodes和max_step_per_episode将失效
        """
        self.q_network.to(self.device)
        self.target_network.to(self.device) 
        
        self.monitor = RewardMonitor(
            max_step_per_episode=max_step_per_episode,
            max_total_steps=max_total_steps,
            target_episode_reward=target_episode_reward,
            target_window_avg_reward=target_window_avg_reward,
            target_window_length=target_window_length
        )
        
        should_stop = False
        for episode_idx in range(num_episodes):
            state, _ = self.env.reset(seed=seed)
            self.monitor.before_episode_start()
            
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.monitor.after_step_env(next_state, reward, terminated, truncated, info)
                
                done = terminated or truncated
                self.memory.add(state, action, reward, next_state, done)
                self.update()
                
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
            if should_stop or (episode_idx + 1) % self.config['verbose_freq'] == 0:
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
    
    def save(self, path):
        # Save the model | 保存模型
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'update_steps': self.update_steps
        }, path)
    
    def load(self, path):
        # Load the model | 加载模型
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self._initialize_components()
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_steps = checkpoint['update_steps']
