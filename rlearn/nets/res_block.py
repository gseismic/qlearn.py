import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_features, out_features, use_batch_norm=True, activation=nn.ReLU()):
        super(ResBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        self.fc1 = nn.Linear(in_features, out_features)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm1d(out_features)
        
        # 确保输入和输出的形状一致
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features) if self.use_batch_norm else nn.Identity()
            )

    def forward(self, x):
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
