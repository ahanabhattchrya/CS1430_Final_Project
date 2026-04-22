import torch
import torch.nn as nn
from blocks import ResidualBlock



class Encoder(nn.Module):
    # 4x (conv(3x3) -> batchnorm -> relu)
    # each stage downsample feature maps by 2x
    def __init__(self, in_chan):
        super().__init__()
        self.in_chan = in_chan
        self.layer1 = nn.Sequential(nn.Conv2d(self.in_chan, 64, 3, stride=2, padding=1), 
                                    nn.BatchNorm2d(64), 
                                    nn.ReLU(),
                                    ResidualBlock(64))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), 
                                    nn.BatchNorm2d(128), 
                                    nn.ReLU(), 
                                    ResidualBlock(128))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), 
                                    nn.BatchNorm2d(256), 
                                    nn.ReLU(), 
                                    ResidualBlock(256))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), 
                                    nn.BatchNorm2d(512), 
                                    nn.ReLU(), 
                                    ResidualBlock(512))
    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        
        return [feat1, feat2, feat3, feat4]


class Decoder(nn.Module):
    # transposed conv and attention block
    ...



class LightShedAE(nn.Module):
    def __init__(self):
        
        self.encoder = Encoder()
        self.bottleneck = ... # conv layer -> 2x residual blocks
        self.decoder = Decoder()
    
    def foward(self, x):
        features = self.encoder(x)
        z = ... # self.bottleneck()
        return self.decoder(z, features)