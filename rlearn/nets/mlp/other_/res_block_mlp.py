import torch
import torch.nn as nn


class ResBlockMLP(nn.Module):
    """
    残差块 | Residual Block
    """

    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 use_batch_norm: bool = True, 
                 activation: nn.Module = nn.ReLU()):
        super(ResBlockMLP, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # in_features -> out_features -> out_features
        self.fc1 = nn.Linear(in_features, out_features)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm1d(out_features)
        
        # 确保输入和输出的形状一致 | Ensure the shape of input and output is consistent
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features) if self.use_batch_norm else nn.Identity()
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # 𝛙ᵢ = 𝛄𝛘̂ᵢ + 𝛃
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        out += identity
        out = self.activation(out)
        return out
