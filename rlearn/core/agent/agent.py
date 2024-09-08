from abc import ABC, abstractmethod
from ...logger import user_logger


class Agent(ABC):

    def __init__(self, name, env, verbose=1, logger=None):
        self.name = name
        self.env = env
        self.verbose = verbose
        self.logger = logger or user_logger

    @abstractmethod
    def predict(self, state, *args, **kwargs):
        pass


class OffPolicyAgent(Agent):
    def __init__(self, name, env, verbose=1, logger=None):
        super().__init__(name, env, verbose, logger)
        self.replay_buffer = ReplayBuffer(capacity=10000)
    
    def learn(self, *args, **kwargs):
        raise NotImplementedError


class OnPolicyAgent(Agent):
    def __init__(self, name, env, verbose=1, logger=None):
        super().__init__(name, env, verbose, logger)
    
    def learn(self, *args, **kwargs):
        raise NotImplementedError
