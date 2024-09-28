import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4, factorized: bool = True, rank: int = 0):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.factorized = factorized
        self.rank = rank if factorized else 0
        self.noise_scale: float = 1.0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        if self.factorized:
            if self.rank > 0:
                self.weight_epsilon_input = nn.Parameter(torch.empty(in_features, rank))
                self.weight_epsilon_output = nn.Parameter(torch.empty(out_features, rank))
            else:
                self.register_buffer('weight_epsilon_input', torch.empty(in_features))
                self.register_buffer('weight_epsilon_output', torch.empty(out_features))
            
            self.register_buffer('bias_epsilon_input', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self) -> None:
        if self.factorized:
            if self.rank > 0:
                epsilon_in = torch.randn(self.in_features, self.rank)
                epsilon_out = torch.randn(self.out_features, self.rank)
                self.weight_epsilon.copy_(torch.matmul(epsilon_out, epsilon_in.t()))
            else:
                epsilon_in = torch.randn(self.in_features)
                epsilon_out = torch.randn(self.out_features)
                self.weight_epsilon.copy_(epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0))
            self.bias_epsilon.copy_(torch.randn(self.out_features))
        else:
            self.weight_epsilon.normal_()
            self.bias_epsilon.normal_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(input, 
                            self.weight_mu + self.noise_scale * self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.noise_scale * self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

    def set_noise_scale(self, scale: float) -> None:
        self.noise_scale = scale

__all__ = ['NoisyLinear']