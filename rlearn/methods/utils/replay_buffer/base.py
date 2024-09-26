
from abc import ABC, abstractmethod
from collections import namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer(ABC):
    @abstractmethod
    def add(self, *args):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def __len__(self):
        pass