import torch
import torch.nn as nn
from typing import List
from .res_block_mlp import ResBlockMLP


class ResMLP(nn.Module):
    """
    Residual Multi-Layer Perceptron Network
    """

    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_sizes: List[int], 
                 hidden_activation: nn.Module = nn.ReLU(), 
                 output_activation: nn.Module = None, 
                 use_batch_norm: bool = False):
        """
        Args:
            input_size: 输入维度 | Input Dimension
            output_size: 输出维度 | Output Dimension
            hidden_sizes: 隐藏层维度列表 | Hidden Layer Dimension List
            hidden_activation: 隐藏层激活函数 | Hidden Layer Activation Function
            output_activation: 输出层激活函数 | Output Layer Activation Function
            use_batch_norm: 是否使用批量归一化 | Whether to use batch normalization
        """
        super(ResMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        layers = []
        in_features = input_size
        
        # 构建残差块 | Build Residual Blocks
        for hidden_size in hidden_sizes:
            layers.append(ResBlockMLP(in_features, hidden_size, use_batch_norm, hidden_activation))
            in_features = hidden_size
        
        # 最后一层 | Last Layer
        self.fc_out = nn.Linear(in_features, output_size)
        self.output_activation = output_activation
        
        # 使用 Sequential 组装模型 | Use Sequential to Assemble Model
        self.model = nn.Sequential(*layers)
        
        # 初始化权重 | Initialize Weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化权重 | Initialize Weights
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 | Forward
        Args:
            x: 输入 | Input
        Returns:
            x: 输出 | Output
        """
        x = self.model(x)
        x = self.fc_out(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x
