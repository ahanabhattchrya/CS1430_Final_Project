import torch
import torch.nn as nn


class EncBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1), 
                                 nn.BatchNorm2d(out_chan), 
                                 nn.ReLU(), 
                                 ResidualBlock(out_chan))
    def forward(self, x):
        return self.enc(x)
    

class DecBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.dec = nn.Sequential(ResidualBlock(in_chan), 
                                 nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2))
    def forward(self, x):
        return self.dec(x)



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
    def __init__(self, in_chan, gating):
        self.in_chan = in_chan
        self.gating = gating
    
    def forward(self, x):
        ...