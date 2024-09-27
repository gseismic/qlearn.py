import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np

class BaseNoisyLinear(nn.Module):
    """
    BaseNoisyLinear 是 NoisyLinear 和 FactorisedNoisyLinear 的基类。
    它实现了两个类共有的方法和属性。
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 init_method: str = 'kaiming',
                 std_init: float = 0.5,
                 exploration_factor: float = 1.0):
        super(BaseNoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.init_method = init_method
        self.exploration_factor = exploration_factor
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        # 移除这里的 reset_noise() 调用
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        
        if self.init_method == 'uniform':
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.bias_mu.data.uniform_(-mu_range, mu_range)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.weight_mu)
            nn.init.constant_(self.bias_mu, 0)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.weight_mu, nonlinearity='relu')
            nn.init.constant_(self.bias_mu, 0)
        else:
            raise ValueError(f"未知的初始化方法: {self.init_method}")
        
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul((x.abs().sqrt() * self.exploration_factor).clamp(0, 1))
    
    def reset_noise(self):
        raise NotImplementedError("子类必须实现 reset_noise 方法")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon * self.exploration_factor
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon * self.exploration_factor
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class DenseNoisyLinear(BaseNoisyLinear):
    """
    DenseNoisyLinear class is a linear layer with added noise for exploration in neural networks.
    This noise helps increase exploration during training, thereby improving the model's generalization ability.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 init_method: str = 'kaiming',
                 std_init: float = 0.5,
                 exploration_factor: float = 1.0):
        super().__init__(in_features, out_features, init_method, std_init, exploration_factor)
        self.reset_noise()

    def reset_noise(self):
        # 使用缩放后的噪声
        self.weight_epsilon.copy_(self._scale_noise(self.weight_epsilon.size()))
        self.bias_epsilon.copy_(self._scale_noise(self.bias_epsilon.size()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 需要在程序里reset_noise
            # NOTE: 如果此处没有clone，会报错
            # weight = self.weight_mu + self.weight_sigma * self.weight_epsilon * self.exploration_factor
            # bias = self.bias_mu + self.bias_sigma * self.bias_epsilon * self.exploration_factor
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon.clone() * self.exploration_factor
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon.clone() * self.exploration_factor
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class FactorizedNoisyLinear(BaseNoisyLinear):
    """
    FactorizedNoisyLinear 类实现了使用因子化高斯噪声的线性层。
    这个版本允许使用 k 个随机高斯噪声向量，增加了噪声的复杂性。
    
    _scale_noise方法生成的不是标准正态分布，而是一种经过变换的分布。这种分布可能在某些情况下更有利于探索。
    References:
         - https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 init_method: str = 'kaiming',
                 std_init: float = 0.5,
                 exploration_factor: float = 1.0,
                 k: int = 1):
        super().__init__(in_features, out_features, init_method, std_init, exploration_factor)
        self.k = k  # 确保 k 被设置为实例属性
        self.register_buffer('epsilon_in', torch.FloatTensor(self.k, in_features))
        self.register_buffer('epsilon_out', torch.FloatTensor(self.k, out_features))
        self.reset_noise()  # 在初始化结束时调用 reset_noise

    def reset_noise(self):
        # TODO: check 
        for i in range(self.k):
            self.epsilon_in[i] = self._scale_noise(self.in_features)
            self.epsilon_out[i] = self._scale_noise(self.out_features)

        # 使用 k 个噪声向量的平均值
        epsilon_in = self.epsilon_in.mean(dim=0)
        epsilon_out = self.epsilon_out.mean(dim=0)

        self.weight_epsilon.copy_(epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 移除这里的 reset_noise 调用
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon.clone() * self.exploration_factor
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon.clone() * self.exploration_factor
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
