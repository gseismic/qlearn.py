import random
import numpy as np  
from typing import List, Tuple

class ReplayBuffer:
    # 经验回放缓冲区 | Experience replay buffer
    def __init__(self, capacity: int):
        assert capacity > 0, 'Capacity must be positive'
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
    def push(self, *args):
        # 存储经验 | Store experience
        if len(self.buffer) < self.capacity:
            self.buffer.append(args)
        else:
            self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        # 随机采样经验 | Randomly sample experiences
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        # 返回当前缓冲区大小 | Return current buffer size
        return len(self.buffer)