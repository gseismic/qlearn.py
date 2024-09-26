import random
from collections import namedtuple, deque
from .base import ReplayBuffer, Experience

class RandomReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
