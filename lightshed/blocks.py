import torch
import torch.nn as nn
import torch.nn.functional as F


class EncBlock(nn.Module):
    # conv(3x3) -> batchnorm -> relu -> resblock  (downsize -> stride = 2)
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1), 
                                 nn.BatchNorm2d(out_chan), 
                                 nn.ReLU(), 
                                 ResidualBlock(out_chan))
    def forward(self, x):
        return self.enc(x)
    

class DecBlock(nn.Module):
    # res block -> transpose conv (upsample)
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.dec = nn.Sequential(ResidualBlock(in_chan), 
                                 nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2))
    def forward(self, x):
        return self.dec(x)


class ResidualBlock(nn.Module):
    # 2x (conv(3x3) -> batchnorm -> relu + skip connection)
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
    # (enc feat map, dec signal prev layer) -> attention map
    def __init__(self, in_chan, gate_chan, out_chan):
        super().__init__()
        self.enc_feat = nn.Conv2d(in_chan, out_chan, kernel_size=1) 
        self.gating = nn.Conv2d(gate_chan, out_chan, kernel_size=1)
        self.attn_map = nn.Conv2d(out_chan, 1, kernel_size=1)
       
    def forward(self, x, g):
        enc_feat = self.enc_feat(x) # feat map from encoder layer [8, 64, 256, 32]
        gate_sig = self.gating(g) # gating signal from decoder [8, 1024, 16, 16]

        # upsample the gating signal for same spatial dim
        up_gate_sig = F.interpolate(gate_sig, size=enc_feat.shape[2:], mode='bilinear',)
        
        combined = enc_feat + up_gate_sig # combine features
        # print('combined', combined.shape)

        attn_map = torch.sigmoid(self.attn_map(combined)) # create attention map [0, 1] important vs not
 
        return attn_map