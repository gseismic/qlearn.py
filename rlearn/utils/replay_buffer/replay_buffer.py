import random
import numpy as np  
from typing import List, Tuple

class ReplayBuffer:
    # 经验回放缓冲区 | Experience replay buffer
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # 存储经验 | Store experience
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        # 随机采样经验 | Randomly sample experiences
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        # 返回当前缓冲区大小 | Return current buffer size
        return len(self.buffer)