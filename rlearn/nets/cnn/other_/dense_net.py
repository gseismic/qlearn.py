import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # shape: [batch_size, in_channels, height, width]
        out = self.conv1(F.relu(self.bn1(x)))
        # shape: [batch_size, 4*growth_rate, height, width]
        out = self.conv2(F.relu(self.bn2(out)))
        # shape: [batch_size, growth_rate, height, width]
        out = torch.cat([out, x], 1)
        # shape: [batch_size, in_channels + growth_rate, height, width]
        return out


class DenseNet(nn.Module):
    def __init__(self, input_channels: int, growth_rate: int, num_blocks: List[int], num_classes: int):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.in_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(num_blocks[0])
        self.trans1 = self._make_transition(self.in_channels)
        self.dense2 = self._make_dense_layers(num_blocks[1])
        self.trans2 = self._make_transition(self.in_channels)
        self.dense3 = self._make_dense_layers(num_blocks[2])
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.linear = nn.Linear(self.in_channels, num_classes)

    def _make_dense_layers(self, num_blocks: int):
        layers = []
        for _ in range(num_blocks):
            layers.append(DenseLayer(self.in_channels, self.growth_rate))
            self.in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels: int):
        out_channels = in_channels // 2
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        # shape: [batch_size, input_channels, height, width]
        out = self.conv1(x)
        # shape: [batch_size, 2*growth_rate, height, width]
        out = self.trans1(self.dense1(out))
        # shape: [batch_size, in_channels//2, height/2, width/2]
        out = self.trans2(self.dense2(out))
        # shape: [batch_size, in_channels//2, height/4, width/4]
        out = self.dense3(out)
        # shape: [batch_size, in_channels, height/4, width/4]
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
        # shape: [batch_size, in_channels, 1, 1]
        out = out.view(out.size(0), -1)
        # shape: [batch_size, in_channels]
        out = self.linear(out)
        # shape: [batch_size, num_classes]
        return out