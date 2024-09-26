import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod

class DQNBase(nn.Module, ABC):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def forward(self, state):
        pass

class DQN(DQNBase):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        # state shape: (batch_size, state_dim)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # shape: (batch_size, action_dim)


class DuelingDQN(DQNBase):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_dim)

    def forward(self, state):
        # state shape: (batch_size, state_dim)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)  # shape: (batch_size, 1)
        advantage = self.advantage_stream(x)  # shape: (batch_size, action_dim)
        return value + advantage - advantage.mean(dim=1, keepdim=True)  # shape: (batch_size, action_dim)
