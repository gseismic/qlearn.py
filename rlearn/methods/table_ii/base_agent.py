import torch
import numpy as np
from collections import deque
from cfgdict import make_config
from rlearn.methods.utils.monitor import RewardMonitor
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
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.monitor = RewardMonitor(
            max_step_per_episode=max_step_per_episode,
            max_total_steps=max_total_steps,
            target_episode_reward=target_episode_reward,
            target_window_avg_reward=target_window_avg_reward,
            target_window_length=target_window_length
        )
        
        should_stop = False
        for episode_idx in range(num_episodes):
            trajectory = deque(maxlen=self.config['n_step'])
            state, _ = self.env.reset(seed=seed)
            self.monitor.before_episode_start()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.monitor.after_step_env(next_state, reward, terminated, truncated, info)
                
                done = terminated or truncated
                trajectory.append((state, action, reward))
                self.update(trajectory=trajectory)
                
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
                
                if done and len(trajectory) > 0:
                    while len(trajectory) > 0:
                        self.update(trajectory=trajectory)  # 使用部分回报 | Use partial return
                        trajectory.popleft()  # 移除轨迹中的第一个状态 | Remove the first state from the trajectory

            self.monitor.after_episode_end()
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
            
    def save(self, file_path):
        raise NotImplementedError("This method should be overridden by subclasses")

    def load(self, file_path):
        raise NotImplementedError("This method should be overridden by subclasses")