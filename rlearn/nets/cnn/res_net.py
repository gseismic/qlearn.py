import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # shape: [batch_size, in_channels, height, width]
        out = F.relu(self.bn1(self.conv1(x)))
        # shape: [batch_size, out_channels, height/stride, width/stride]
        out = self.bn2(self.conv2(out))
        # shape: [batch_size, out_channels, height/stride, width/stride]
        out += self.shortcut(x)
        out = F.relu(out)
        # shape: [batch_size, out_channels, height/stride, width/stride]
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels: int, num_blocks: List[int], num_classes: int):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # shape: [batch_size, input_channels, height, width]
        out = F.relu(self.bn1(self.conv1(x)))
        # shape: [batch_size, 64, height, width]
        out = self.layer1(out)
        # shape: [batch_size, 64, height, width]
        out = self.layer2(out)
        # shape: [batch_size, 128, height/2, width/2]
        out = self.layer3(out)
        # shape: [batch_size, 256, height/4, width/4]
        out = self.layer4(out)
        # shape: [batch_size, 512, height/8, width/8]
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # shape: [batch_size, 512, 1, 1]
        out = out.view(out.size(0), -1)
        # shape: [batch_size, 512]
        out = self.linear(out)
        # shape: [batch_size, num_classes]
        return out
