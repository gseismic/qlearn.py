import torch
from abc import ABC, abstractmethod
from cfgdict import make_config
from rlearn.logger import user_logger
from rlearn.methods.utils.monitor import RewardMonitor

class BaseDQNAgent(ABC):

    def __init__(self, env, config=None, logger=None):
        self.env = env
        self.logger = logger or user_logger
        self.config = make_config(config, schema=self.schema,
                                  logger=self.logger,
                                  to_dict=True,
                                  to_dict_flatten=True)
        self.monitor = None
        if self.config['device'] == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f'Config: {self.config}')
        # You should call init_networks() in the subclass
        # self.init_networks()

    @abstractmethod
    def init_networks(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def learn(self, 
              num_episodes, 
              max_step_per_episode=None, 
              max_total_steps=None, 
              target_episode_reward=None, 
              target_window_avg_reward=None,
              target_window_length=None,
              seed=None, **kwargs):
       pass
   
    def save(self, file_path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
