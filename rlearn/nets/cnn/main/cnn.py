import torch
import torch.nn as nn
from typing import List, Tuple
from ...utils.initializer import get_initializer
from ...utils.activation import get_activation

class CNN(nn.Module):
    def __init__(self, input_channels: int, 
                 conv_layers: List[Tuple[int, int, int]], 
                 activation: str = 'relu', 
                 init_type: str = 'kaiming_uniform'):
        super(CNN, self).__init__()
        self.activation = get_activation(activation)
        self.initializer = get_initializer(init_type)

        layers: List[nn.Module] = []
        in_channels = input_channels    
        for out_channels, kernel_size, stride in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            layers.append(self.activation)
            in_channels = out_channels

        self.conv_net = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            self.initializer(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (batch_size, input_channels, height, width)
        # output: shape (batch_size, out_channels, out_height, out_width)
        return self.conv_net(x)
    
__all__ = ['CNN']