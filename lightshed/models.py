import torch
import torch.nn as nn
from blocks import ResidualBlock, AttentionBlock, EncBlock, DecBlock
import torch.nn.functional as F



class Encoder(nn.Module):
    # 4x (enc block) -> features 1-4
    # each stage downsample feature maps by 2x
    def __init__(self, in_chan):
        super().__init__()
        self.in_chan = in_chan # 3
        self.layer1 = EncBlock(self.in_chan, 64)
        self.layer2 = EncBlock(64, 128)
        self.layer3 = EncBlock(128, 256)
        self.layer4 = EncBlock(256, 512)

    def forward(self, x):
        feat1 = self.layer1(x) # [8, 64, 256, 256]
        # print('f1', feat1.shape) 
        feat2 = self.layer2(feat1) # [8, 64, 256, 128]
        # print('f2', feat2.shape)
        feat3 = self.layer3(feat2) # [8, 64, 256, 64]
        # print('f3', feat3.shape)
        feat4 = self.layer4(feat3) # [8, 64, 256, 32]
        # print('f4', feat4.shape)

        return [feat1, feat2, feat3, feat4]


class Decoder(nn.Module):
    # 4x (dec block w/ attention) -> reconstruction
    def __init__(self, bottle_out):
        super().__init__()
        self.bottle_out = bottle_out # output dim of bottleneck layer (in_chan for first dec block)

                            # [in_chan, out_chan]
        self.dec4 = DecBlock(self.bottle_out, 512) #[bottleneck out = 1024, enc feat4 out_chan  = 512]
        self.dec3 = DecBlock(512, 256) # [prev dec out_chan = 512, enc feat3 out_chan = 256]
        self.dec2 = DecBlock(256, 128) # [prev dec out_chan = 256, enc feat2 out_chan = 128]
        self.dec1 = DecBlock(128, 64) # [prev dec out_chan = 128, enc feat1 out_chan = 64]

                                # [in_chan, gate_chan, out_chan]
        self.attn4 = AttentionBlock(512, self.bottle_out, 256) # [enc feat4 = 512, dec4 in/bottleneck out = 1024]
        self.attn3 = AttentionBlock(256, 512, 128) # [enc feat3 = 256, dec4 output = 512]
        self.attn2 = AttentionBlock(128, 256, 64) # [enc feat2 = 128, dec3 output = 256]
        self.attn1 = AttentionBlock(64, 128, 32)  # [enc feat1 = 64, dec2 output = 128]

        self.out = nn.Conv2d(64, 3, kernel_size=1) # [dec out_chan = 64, out img=3], kernel size ?
    
    def forward(self, bottle_out, features):
        # feat_map from corresponding encoder layer
        f1, f2, f3, f4 = features

        # For dec layer [i]
        # 1. out[i] = self.dec[i](out_layer[i-1])
        # 2. attn[i] = self.attn[i](enc_feat[i], out_layer[i-1])
        # 3. out[i] w/ attn = out[i] * attn[i] 

        ###### block 4 ######

        d4 = self.dec4(bottle_out)
        # print('d4', d4.shape)
        # print('f4', f4.shape)
        # print('b', bottle_out.shape)
        # f4 is 32, while bottle out is 16 because bottle out further downsamples the f4 
        # need to upsample bottle out before 
        attn4 = self.attn4(f4, bottle_out)
        # print('att4', attn4.shape)
        d4 = d4 * attn4
        # print('d4', d4.shape)
        

        ###### block 3 ######
        d3 = self.dec3(d4)
        attn3 = self.attn3(f3, d4)
        # print('att3', attn3.shape)
        
        d3 = d3 * attn3
        # print('d3', d3.shape)

        ###### block 2 ######
        d2 = self.dec2(d3)
        attn2 = self.attn2(f2, d3)
        # print('att2', attn2.shape)
        d2 = d2 * attn2
        # print('d2', d2.shape)

        ###### block 1 ######
        d1 = self.dec1(d2)
        attn1 = self.attn1(f1, d2)
        d1 = d1 * attn1
        # print('d1', d1.shape)
        # one more upsample to align with ground truth images
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear')
        
        return self.out(d1)

    
class LightShedAE(nn.Module):
    def __init__(self, in_chan=3):
        super().__init__()
        
        # downsample 
        self.encoder = Encoder(in_chan)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(1024), 
                                        nn.ReLU(), 
                                        ResidualBlock(1024), 
                                        ResidualBlock(1024)) # conv layer -> 2x residual blocks
        # upsample + attention
        # in_chan = bottleneck out_chan
        self.decoder = Decoder(1024)
    
    def forward(self, x):
        features = self.encoder(x) #[f1, f2, f3, f4]
        z = self.bottleneck(features[-1])
        return self.decoder(z, features)