from abc import ABC, abstractmethod

class ExitMonitor(ABC):
    """
    退出监控器
    """
    def __init__(self):
        self._should_exit = False
        self._exit_code = None
        self._exit_info = None
    
    @abstractmethod
    def check_exit(self, *args, **kwargs):
        pass

    def set_exit(self, exit_code, exit_info):
        self._should_exit = True 
        self._exit_code = exit_code
        self._exit_info = exit_info

    @property
    def should_exit(self):
        return self._should_exit

    @property
    def exit_code(self):
        return self._exit_code
    
    @property
    def exit_info(self):
        return self._exit_info
    
class RewardMonitor(ExitMonitor):
    """
    奖励监控器
    """
    def __init__(self, target_reward, target_avg_reward, avg_type, avg_period):
        super().__init__()
        self.target_reward = target_reward
        self.reward_list = []
        self.average_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        
    def check_exit(self, episode_idx, episode_istep, episode_reward, episode_done, step_reward):
        if episode_done:
            self.reward_list.append(episode_reward)
        if episode_reward >= self.target_reward:
            exit_info = {'average_reward': self.get_average_reward()}
            self.set_exit(0, exit_info)

    def get_average_reward(self):
        return sum(self.reward_list) / len(self.reward_list)
    

class AverageRewardMonitor(ExitMonitor):
    """
    奖励监控器
    """
    def __init__(self, target_avg_reward, avg_type, avg_period):
        super().__init__()
        self.target_avg_reward = target_avg_reward
        self.avg_type = avg_type
        self.avg_period = avg_period
        
        self.target_reward = target_reward
        self.reward_list = []
        self.average_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        
    def check_exit(self, episode_idx, episode_istep, episode_reward, episode_done, step_reward):
        if episode_done:
            self.reward_list.append(episode_reward)
        if episode_reward >= self.target_reward:
            exit_info = {'average_reward': self.get_average_reward()}
            self.set_exit(0, exit_info)

    def get_average_reward(self):
        return sum(self.reward_list) / len(self.reward_list)