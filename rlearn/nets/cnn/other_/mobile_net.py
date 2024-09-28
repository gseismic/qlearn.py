import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # shape: [batch_size, in_channels, height, width]
        out = self.depthwise(x)
        # shape: [batch_size, in_channels, height/stride, width/stride]
        out = self.pointwise(out)
        # shape: [batch_size, out_channels, height/stride, width/stride]
        return out

class MobileNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 512, stride=2),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024)
        )
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        # shape: [batch_size, input_channels, height, width]
        out = F.relu(self.bn1(self.conv1(x)))
        # shape: [batch_size, 32, height/2, width/2]
        out = self.layers(out)
        # shape: [batch_size, 1024, height/32, width/32]
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # shape: [batch_size, 1024, 1, 1]
        out = out.view(out.size(0), -1)
        # shape: [batch_size, 1024]
        out = self.linear(out)
        # shape: [batch_size, num_classes]
        return out