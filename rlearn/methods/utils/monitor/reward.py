from collections import deque
from .base import BaseMonitor

class RewardMonitor(BaseMonitor):
    name = 'reward-monitor'
    def __init__(self, 
                 max_step_per_episode=None, 
                 max_total_steps=None, 
                 target_episode_reward=None,
                 target_window_avg_reward=None,
                 target_window_length=None):
        self.max_step_per_episode = max_step_per_episode
        self.max_total_steps = max_total_steps
        self.target_episode_reward = target_episode_reward
        self.target_window_avg_reward = target_window_avg_reward
        self.target_window_length = target_window_length
        self.total_steps = 0
        self.episode_steps = 0
        self.episode_reward = 0
        self.all_episode_rewards = []
        self.window_episode_rewards = deque(maxlen=target_window_length)

    def before_episode_start(self):
        self.episode_steps = 0
        self.episode_reward = 0
        self.total_steps = 0
        
    def after_env_step(self, next_state, reward, terminated, truncated, info):
        self.episode_steps += 1
        self.episode_reward += reward
        self.total_steps += 1
    
    def after_episode_end(self):
        self.window_episode_rewards.append(self.episode_reward)
        self.all_episode_rewards.append(self.episode_reward)
        
    def check_exit_conditions(self):
        exit_episode = False
        if self.max_step_per_episode is not None and self.episode_steps >= self.max_step_per_episode:
            exit_episode = True
        
        exit_learning = False
        exit_learning_code = -1
        exit_learning_msg = 'DO NOT EXIT'
        if self.target_episode_reward is not None and self.episode_reward >= self.target_episode_reward:
            exit_learning = True
            exit_learning_code = 0
            exit_learning_msg = f"Reached target reward: {self.target_episode_reward}"
        elif self.max_total_steps is not None and self.total_steps >= self.max_total_steps:
            exit_learning = True
            exit_learning_code = 1
            exit_learning_msg = f"Reached max total steps: {self.max_total_steps}"
        elif self.target_window_avg_reward is not None and len(self.window_episode_rewards) >= self.target_window_length:
            avg_reward = sum(self.window_episode_rewards) / len(self.window_episode_rewards)
            if avg_reward >= self.target_window_avg_reward:
                exit_learning = True
                exit_learning_code = 2
                exit_learning_msg = f"Reached target window avg reward: {self.target_window_avg_reward}"
        return exit_episode, exit_learning, (exit_learning_code, exit_learning_msg)

    def get_exit_info(self):
        info = {
            'total_steps': self.total_steps,
            'last_episode_steps': self.episode_steps,
            'last_episode_reward': self.episode_reward,
            'all_episode_rewards': self.all_episode_rewards,
            'max_step_per_episode': self.max_step_per_episode,
            'max_total_steps': self.max_total_steps,
            'target_episode_reward': self.target_episode_reward,
            'target_window_avg_reward': self.target_window_avg_reward,
            'target_window_length': self.target_window_length
        }
        return info