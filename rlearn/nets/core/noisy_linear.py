import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np

class NoisyLinear(nn.Module):
    """
    NoisyLinear 类是用于在神经网络中添加噪声的线性层。| NoisyLinear is a linear layer with noise added to it for exploration in neural networks.
    这种噪声有助于在训练过程中增加探索性，从而提高模型的泛化能力。| This noise helps increase exploration during training, thereby improving the model's generalization ability.
    
    Notes:
        - 参数量加倍，计算量加倍了，可以考虑低秩分解版本噪音 | parameter doubling, computational cost doubling, consider low-rank decomposition version noise
        - 可应用于DQN，PG，AC，A2C，PPO等算法，增强探索性 | can be applied to DQN, PG, AC, A2C, PPO and other algorithms, enhancing exploration
    ---
    Reference:
        - [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 init_method: str = 'kaiming',
                 std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.init_method = init_method
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # weight_epsilon: (out_features, in_features) 用于探索 | used for exploration
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        # 重置所有参数 | Reset all parameters
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
            raise ValueError(f"未知的初始化方法 | Unknown initialization method: {self.init_method}")
        
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        # 重置噪声 | Reset noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        # 缩放噪声 | Scale noise
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 前向传播 | Forward pass
        # input: (batch_size, in_features) -> (batch_size, out_features)
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
