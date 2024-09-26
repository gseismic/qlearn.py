import torch
import numpy as np
from collections import deque
from cfgdict import make_config
from rlearn.logger import user_logger
# from rlearn.methods.table_ii.base_agent import BaseAgent
from rlearn.methods.table_ii.monitor import Monitor

class NStepQAgent:
    """
    N-Step Q-Learning Agent
    """
    schema = [
        dict(field='n_step', required=False, default=3, rules=dict(type='int', gt=0)),
        dict(field='use_strict_n_step', required=False, default=False, rules=dict(type='bool')),
        dict(field='learning_rate', required=False, default=0.1, rules=dict(type='float', gt=0, max=1)),
        dict(field='gamma', required=False, default=0.99, rule=dict(type='float', min=0, max=1)),
        dict(field='epsilon', required=False, default=0.1, rules=dict(type='float', min=0, max=1)),
        dict(field='verbose_freq', required=False, default=10, rules=dict(type='int', gt=0)),
    ]
    def __init__(self, env, config=None, logger=None):
        # super().__init__(env, config, logger, monitor)
        self.logger = logger or user_logger
        self.config = make_config(config, schema=self.schema,
                                  logger=self.logger,
                                  to_dict=True,
                                  to_dict_flatten=True)
        self.env = env
        self.monitor = None

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q_table = torch.zeros(self.n_states, self.n_actions)
        self.gamma = self.config['gamma']
        self.logger.info(f'Config: {self.config}')
        
    def select_action(self, state):
        if np.random.rand() < self.config['epsilon']:
            return np.random.choice(self.n_actions)
        else:
            return torch.argmax(self.Q_table[state]).item()

    def update(self, trajectory):
        if self.config['use_strict_n_step'] is True and len(trajectory) < self.config['n_step']:
            return False
        
        current_n_step = len(trajectory)
        assert 0 < current_n_step <= self.config['n_step']
        G = 0
        for i, (state, action, reward) in enumerate(trajectory):
            G += (self.gamma ** i) * reward
        
        state, action, _ = trajectory[0]
        next_state, _, _ = trajectory[-1]

        G += (self.gamma ** current_n_step) * torch.max(self.Q_table[next_state]).item()
        td_error = G - self.Q_table[state][action]
        self.Q_table[state][action] += self.config['learning_rate'] * td_error
        return True

    def learn(self, 
              num_episodes, 
              max_step_per_episode=None, 
              max_total_steps=None, 
              target_reward=None, 
              seed=None):
        """
        Args: 
            - num_episodes: 训练的次数 | Number of episodes to train
            - max_step_per_episode: 每个episode的最大步数 | Maximum steps per episode
            - max_total_steps: 总的训练步数 | Total steps to train
            - target_reward: 目标奖励 | Target reward to achieve
            - seed: 随机种子 | Random seed
        Returns: None
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.monitor = Monitor(max_step_per_episode=max_step_per_episode,
                               max_total_steps=max_total_steps,
                               target_reward=target_reward)
        
        for episode in range(num_episodes):
            trajectory = deque(maxlen=self.config['n_step'])
            self.monitor.reset()
            state, _ = self.env.reset()
            
            action = self.select_action(state)
            done = False
            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                next_action = self.select_action(next_state)
                self.monitor.update(next_state, reward, terminated, truncated, info)
                trajectory.append((state, action, reward))
                self.update(trajectory=trajectory)
                
                state = next_state
                action = next_action
                exit_episode, exit_learning, exit_learning_msg = self.monitor.check_exit_conditions()
                if exit_episode:
                    break
                
                if exit_learning:
                    self.logger.info(exit_learning_msg)
                    return
                
                if done and len(trajectory) > 0:
                    while len(trajectory) > 0:
                        self.update(trajectory=trajectory)  # 使用部分回报 | Use partial return
                        trajectory.popleft()  # 移除轨迹中的第一个状态 | Remove the first state from the trajectory

            if (episode + 1) % self.config['verbose_freq'] == 0:
                self.logger.info(f"Episode {episode+1}/{num_episodes}, Total Reward: {self.monitor.total_reward}")

    def save(self, file_path):
        torch.save(self.Q_table, file_path)
        self.logger.info(f"Q-table saved to {file_path}")

    def load(self, file_path):
        self.Q_table = torch.load(file_path)
        self.logger.info(f"Q-table loaded from {file_path}")