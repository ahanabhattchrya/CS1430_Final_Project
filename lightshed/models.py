import torch
import torch.nn as nn
from blocks import ResidualBlock, AttentionBlock, EncBlock, DecBlock



class Encoder(nn.Module):
    # 4x (conv(3x3) -> batchnorm -> relu)
    # each stage downsample feature maps by 2x
    def __init__(self, in_chan):
        super().__init__()
        self.in_chan = in_chan # 3
        self.layer1 = EncBlock(self.in_chan, 64)
        self.layer2 = EncBlock(64, 128)

        self.layer3 = EncBlock(128, 265)
        self.layer4 = EncBlock(256, 512)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        
        return [feat1, feat2, feat3, feat4]


class Decoder(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.in_chan = in_chan
        self.dec1 = DecBlock(self.in_chan, )
        self.dec2 = DecBlock()
        self.dec2 = DecBlock()
        self.dec2 = DecBlock()
        # transposed conv and attention block
    
    def forward(self, bottle_out, features):
        # feat_map from corresponding encoeer layer * attention map
        f1, f2, f3, f4 = features
        d1 = self.dec1(bottle_out)
        



class LightShedAE(nn.Module):
    def __init__(self):
        
        self.encoder = Encoder()
        self.bottleneck = nn.Sequential(nn.Conv2d(3, 512, 3, stride=2), 
                                        ResidualBlock(512), 
                                        ResidualBlock(512)) # conv layer -> 2x residual blocks
        self.decoder = Decoder()
    
    def foward(self, x):
        features = self.encoder(x)
        z = ... # self.bottleneck()
        return self.decoder(z, features)