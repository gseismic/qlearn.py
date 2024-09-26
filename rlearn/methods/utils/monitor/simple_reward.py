from .base import BaseMonitor

class SimpleRewardMonitor(BaseMonitor):
    def __init__(self, max_step_per_episode=None, max_total_steps=None, target_reward=None):
        self.total_steps = 0
        self.total_reward = 0
        self.episode_steps = 0
        self.max_step_per_episode = max_step_per_episode
        self.max_total_steps = max_total_steps
        self.target_reward = target_reward

    def reset(self):
        self.total_reward = 0
        self.episode_steps = 0

    def update(self, next_state, reward, terminated, truncated, info):
        self.total_reward += reward
        self.episode_steps += 1
        self.total_steps += 1

    def check_exit_conditions(self):
        exit_episode = False    
        if self.max_step_per_episode and self.episode_steps >= self.max_step_per_episode:
            exit_episode = True
        
        exit_learning = False
        exit_learning_msg = None
        if self.target_reward and self.total_reward >= self.target_reward:
            exit_learning = True
            exit_learning_msg = f"Reached target reward: {self.total_reward}"
        elif self.max_total_steps and self.total_steps >= self.max_total_steps:
            exit_learning = True
            exit_learning_msg = f"Reached max total steps: {self.total_steps}"
                
        return exit_episode, exit_learning, exit_learning_msg

    def get_monitor_info(self):
        return {
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'episode_steps': self.episode_steps,
            'max_step_per_episode': self.max_step_per_episode,
            'max_total_steps': self.max_total_steps,
            'target_reward': self.target_reward}