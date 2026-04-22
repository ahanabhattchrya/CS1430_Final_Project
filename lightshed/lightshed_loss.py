
import torch.nn as nn

class LightShedLoss(nn.Module):
    ...
    def forward(self, P_recon, P_actual, epoch):
        # p recon is reconstructed perturbation
        # p actual is ground truth perturbation
        ...
    # equations 1-5 in section 4.3 (poison reconstruction)