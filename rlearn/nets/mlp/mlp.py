import torch
import torch.nn as nn
from typing import List
from ..utils.weight_init import init_linear_weight

class MLP(nn.Module):
    """
    Multi-Layer Perceptron Network
    """
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_sizes: List[int], 
                 hidden_activation: nn.Module = nn.ReLU(), 
                 output_activation: nn.Module = None, 
                 use_batch_norm: bool = False,
                 init_method: str = 'xavier'):
        """
        Multi-Layer Perceptron Network
        Args:
            input_size: 输入维度 | Input Dimension
            output_size: 输出维度 | Output Dimension
            hidden_sizes: 隐藏层维度列表 | Hidden Layer Dimension List
            hidden_activation: 隐藏层激活函数 | Hidden Layer Activation Function
            output_activation: 输出层激活函数 | Output Layer Activation Function
            use_batch_norm: 是否使用批量归一化 | Whether to use batch normalization
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.init_method = init_method
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(hidden_activation)
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        if output_activation is not None:
            layers.append(output_activation)
        
        self.network = nn.Sequential(*layers)
        self.apply(lambda module: init_linear_weight(module=module, init_method=self.init_method))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

