from abc import ABC, abstractmethod
from cfgdict import make_config
from rlearn.logger import user_logger
from rlearn.utils.seed import seed_all

class BaseAgent(ABC):
    schema = []

    def __init__(self, env, config=None, logger=None, seed=None):
        self.env = env
        self.logger = logger or user_logger
        self.config = make_config(config, 
                                  schema=self.schema,
                                  logger=self.logger,
                                  to_dict=True,
                                  to_dict_flatten=False)
        self.seed_all(seed)
        self.monitor = None 
        self.init()
    
    def set_monitor(self, monitor):
        self.monitor = monitor
    
    def seed_all(self, seed):
        self.seed = seed
        seed_all(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        
    def init(self):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    