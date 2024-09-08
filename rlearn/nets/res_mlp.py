import torch
import torch.nn as nn
import torch.nn.functional as F
from .res_block import ResBlock


class ResMLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes,
                 use_batch_norm=True, hidden_activation=nn.ReLU(), 
                 output_activation=None):
        super(ResMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        layers = []
        in_features = input_size
        
        # 构建残差块
        for hidden_size in hidden_sizes:
            layers.append(ResBlock(in_features, hidden_size, use_batch_norm, hidden_activation))
            in_features = hidden_size
        
        # 最后一层
        self.fc_out = nn.Linear(in_features, output_size)
        self.output_activation = output_activation
        
        # 使用 Sequential 组装模型
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        x = self.model(x)
        x = self.fc_out(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x
