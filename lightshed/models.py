import torch
import torch.nn as nn



class Encoder(nn.Module):
    # 4x (conv(3x3) -> batchnorm -> relu)
    # each stage downsample feature maps by 2x
    ...


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