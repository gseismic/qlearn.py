from typing import List, Optional
import torch
import torch.nn as nn
from ...utils.module import get_activation, get_initializer
from ...linear.api import get_linear, is_linear, is_noisy_linear

class MLP(nn.Module):
    linear_tag = 'main'
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: str = 'relu', init_type: str = 'kaiming', 
                 use_noisy: bool = False, factorized: bool = True, rank: int = 0, std_init: float = 0.4, noise_level: float = 1.0, output_activation: Optional[str] = None):
        super(MLP, self).__init__()
        self.activation = get_activation(activation)
        self.initializer = get_initializer(init_type)
        self.output_activation = get_activation(output_activation) if output_activation else None

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(get_linear(self.linear_tag, **{
                'in_features': in_dim, 
                'out_features': hidden_dim, 
                'noisy': use_noisy, 
                'factorized': factorized, 
                'rank': rank, 
                'std_init': std_init, 
                'noise_level': noise_level
            }))
            layers.append(self.activation)
            in_dim = hidden_dim
            
        layers.append(get_linear(self.linear_tag, **{
            'in_features': in_dim, 
            'out_features': output_dim, 
            'noisy': use_noisy, 
            'factorized': factorized, 
            'rank': rank, 
            'std_init': std_init, 
            'noise_level': noise_level
        }))
        
        if self.output_activation:
            layers.append(self.output_activation)

        self.network = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if is_linear(m):
            self.initializer(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def reset_noise(self) -> None:
        for module in self.modules():
            if is_noisy_linear(module):
                module.reset_noise()

    def set_noise_scale(self, scale: float) -> None:
        for module in self.modules():
            if is_noisy_linear(module):
                module.set_noise_scale(scale)
