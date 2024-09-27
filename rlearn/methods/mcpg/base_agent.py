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
        pass


