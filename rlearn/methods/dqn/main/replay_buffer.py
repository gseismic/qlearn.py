import random
import numpy as np
from abc import ABC, abstractmethod
from collections import namedtuple, deque

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

class RandomReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, *args):
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5):
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.pos = 0
        self.tree_sum = 0
        self.max_priority = 1.0

    def add(self, *args):
        # Add new experience to the buffer | 将新经验添加到缓冲区
        experience = Experience(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        # Update priorities | 更新优先级
        self.priorities[self.pos] = float(self.max_priority)
        self.tree_sum += self.max_priority ** self.alpha - (self.priorities[self.pos] ** self.alpha if len(self.buffer) == self.capacity else 0)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        # Sample experiences based on priorities | 基于优先级采样经验
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            return [], [], []

        # P(i) = p_i^α / Σ_k p_k^α
        priorities = self.priorities[:buffer_len] ** self.alpha
        probs = priorities / priorities.sum()  # Ensure sum of probabilities is 1

        indices = np.random.choice(buffer_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        self.beta = min(1.0, self.beta + self.beta_increment)
        # w_i = (N * P(i))^(-β) / max_j w_j
        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        # Update priorities based on TD errors | 根据TD误差更新优先级
        for idx, td_error in zip(indices, td_errors):
            td_error_scalar = td_error.item() if hasattr(td_error, 'item') else td_error
            priority = float((abs(td_error_scalar) + self.epsilon) ** self.alpha)
            self.tree_sum += priority - self.priorities[idx] ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)