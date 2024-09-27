import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
from rlearn.nets.core.noisy_linear import DenseNoisyLinear, FactorizedNoisyLinear

class DQNBase(nn.Module, ABC):
    def __init__(self, state_dim, action_dim, use_noisy_net, noisy_net_type, noisy_net_std_init, noisy_net_k, noise_decay, min_exploration_factor):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy_net = use_noisy_net
        self.noisy_net_type = noisy_net_type
        self.noisy_net_std_init = noisy_net_std_init
        self.noisy_net_k = noisy_net_k
        self.noise_decay = noise_decay
        self.min_exploration_factor = min_exploration_factor

    def get_linear(self, in_features, out_features):
        if self.use_noisy_net:
            if self.noisy_net_type == 'dense':
                return DenseNoisyLinear(
                    in_features, 
                    out_features, 
                    std_init=self.noisy_net_std_init, 
                    exploration_factor=1.0, 
                    noise_decay=self.noise_decay,
                    min_exploration_factor=self.min_exploration_factor
                )
            elif self.noisy_net_type == 'factorized':
                return FactorizedNoisyLinear(
                    in_features, 
                    out_features, 
                    std_init=self.noisy_net_std_init, 
                    exploration_factor=1.0, 
                    k=self.noisy_net_k,
                    noise_decay=self.noise_decay,
                    min_exploration_factor=self.min_exploration_factor
                )
        return nn.Linear(in_features, out_features)

    @abstractmethod
    def forward(self, state):
        pass

    def reset_noise(self):
        if self.use_noisy_net:
            for module in self.modules():
                if isinstance(module, (DenseNoisyLinear, FactorizedNoisyLinear)):
                    module.reset_noise()

class DQN(DQNBase):
    def __init__(self, state_dim, action_dim, use_noisy_net, noisy_net_type, noisy_net_std_init, noisy_net_k, min_exploration_factor=0.1, noise_decay=0.99):
        super().__init__(state_dim, action_dim, use_noisy_net, noisy_net_type, noisy_net_std_init, noisy_net_k, noise_decay, min_exploration_factor)
        self.fc1 = self.get_linear(state_dim, 64)
        self.fc2 = self.get_linear(64, 64)
        self.fc3 = self.get_linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(DQNBase):
    def __init__(self, state_dim, action_dim, use_noisy_net, noisy_net_type, noisy_net_std_init, noisy_net_k, min_exploration_factor=0.1, noise_decay=0.99):
        super().__init__(state_dim, action_dim, use_noisy_net, noisy_net_type, noisy_net_std_init, noisy_net_k, noise_decay, min_exploration_factor)
        self.fc1 = self.get_linear(state_dim, 64)
        self.fc2 = self.get_linear(64, 64)
        self.value_stream = self.get_linear(64, 1)
        self.advantage_stream = self.get_linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
