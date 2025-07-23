import torch.nn as nn
import torch.nn.functional as F


# Basic residual block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # Residual path
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # Add residual and shortcut
        return F.relu(self.residual(x) + self.shortcut(x))


# ResNet architecture
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, block_counts=[2, 2, 2, 2]):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self.make_layer(64, block_counts[0], stride=1)
        self.layer2 = self.make_layer(128, block_counts[1], stride=2)
        self.layer3 = self.make_layer(256, block_counts[2], stride=2)
        self.layer4 = self.make_layer(512, block_counts[3], stride=2)

        # Fully connected output
        self.fc1 = nn.Linear(512, out_channels)

    def make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


def resnet18(in_channels, out_channels):
    return ResNet(in_channels, out_channels, [2, 2, 2, 2])
