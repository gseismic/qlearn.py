import torch
import numpy as np
from collections import deque
from cfgdict import make_config
from rlearn.methods.table_ii.monitor import Monitor
from rlearn.logger import user_logger

class BaseAgent:
    schema = []

    def __init__(self, env, config=None, logger=None):
        self.env = env
        self.logger = logger or user_logger
        self.config = make_config(config, schema=self.schema,
                                  logger=self.logger,
                                  to_dict=True,
                                  to_dict_flatten=True)
        self.monitor = None
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q_table = torch.zeros(self.n_states, self.n_actions)
        self.gamma = self.config['gamma']
        self.logger.info(f'Config: {self.config}')

    def select_action(self, state):
        raise NotImplementedError("This method should be overridden by subclasses")

    def update(self, trajectory):
        raise NotImplementedError("This method should be overridden by subclasses")

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
        raise NotImplementedError("This method should be overridden by subclasses")

    def load(self, file_path):
        raise NotImplementedError("This method should be overridden by subclasses")