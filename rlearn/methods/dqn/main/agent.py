import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from .replay_buffer import (
    Experience,
    RandomReplayBuffer, PrioritizedReplayBuffer
)
from ....logger import user_logger
from .network import DQN, DuelingDQN

class DQNAgent_Main:
    """Online DQN Agent
    DQNAgent_Main is a class for training and evaluating a DQN agent.
    
    Notes:
        - Discrete action space Only | 仅支持离散动作空间
    
    reference:
        - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        - https://github.com/Curt-Park/rainbow-is-all-you-need/tree/master?tab=readme-ov-file 
        - https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/dqn.py
    """
    
    DEFAULT_CONFIG = {
        'lr': 1e-3,
        'batch_size': 64,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 10,
        'memory_size': 10000,
        # 'mode': 'online',
        'dueling_dqn': True,
        'double_dqn': True,
        'prioritized_replay': True,
        'hidden_layers': [128, 128],
        'device': 'cpu' # TODO: CUDA-default
    }
    
    def __init__(self, env, config=None):
        """
        Args:
            - env: 环境 | Environment
            - config: 配置 | Configuration
        """
        self.env = env
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.n
        self.logger = user_logger
        self.set_config(config)
        if self.config['device'] == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.epsilon = self.config['epsilon_start']
        self.update_steps = 0
        self.logger.info(f"DQNAgent_Main initialized with state_dim: {self.state_dim}, action_dim: {self.action_dim}")

    def set_config(self, config: dict): 
        """
        Args:
            - config: 配置 | Configuration
        """
        config = config or {}
        self.config = { **self.DEFAULT_CONFIG, **config }
        self.logger.info(f"Config updated: {self.config}")
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化组件 | Initialize or reinitialize components based on current configuration
        """
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
    
    def learn(self, num_episodes, max_step_per_episode=None, max_total_steps=None, target_reward=None, seed=None):
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
        
        reward_list = []
        should_exit = False
        exit_code, exit_info = None, {}
        # exit_monitor = RewardMonitor(target_reward)
        all_episode_steps = 0
        for episode_idx in range(num_episodes):
            if should_exit:
                break
            state, _ = self.env.reset(seed=seed)
            episode_reward = 0
            episode_step = 0
            
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                self.memory.add(state, action, reward, next_state, done)
                # if self.config['mode'] == 'online':
                self.update()
                
                state = next_state
                episode_step += 1
                all_episode_steps += 1
                
                if done or (max_step_per_episode is not None and episode_step >= max_step_per_episode):
                    break
                
                if max_total_steps is not None and self.update_steps >= max_total_steps:
                    self.logger.info(f"Reached total steps of {max_total_steps}")
                    exit_code = 0
                    exit_info = {}
                    should_exit = True
                    break
            
            # if self.config['mode'] == 'offline':
            #     for _ in range(episode_step):
            #         self.update()
            
            self.logger.info(f"Episode {episode_idx + 1}, Reward: {episode_reward}, Epsilon: {self.epsilon:.2f}")
            reward_list.append(episode_reward)
            if target_reward is not None and episode_reward >= target_reward:
                self.logger.info(f"Reached target reward of {target_reward}")
                exit_code = 0
                exit_info = {}
                should_exit = True
                continue
                
        exit_info.update({
            "reward_list": reward_list,
            "update_steps": self.update_steps,
            "all_episode_steps": all_episode_steps
        })
        return exit_code, exit_info
    
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
