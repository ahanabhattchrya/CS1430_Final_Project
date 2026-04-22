import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    # 2x (conv(3x3) -> batchnorm -> relu)
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1), 
                                   nn.BatchNorm2d(channel), 
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1), 
                                   nn.BatchNorm2d(channel))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + x)



class AttentionBlock(nn.Module):
    # 
    ...