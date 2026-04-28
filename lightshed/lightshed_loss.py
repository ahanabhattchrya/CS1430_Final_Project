
import torch.nn as nn

class LightShedLoss(nn.Module):
    def __init__(self, psi=0.01, tau=100):
        super().__init__()
        self.psi = psi
        self.tau = tau
        self.l1 = nn.L1Loss(reduction='none')
    
    
    def forward(self, P_recon, P_actual, is_clean, epoch):
        loss_poison = self.l1(P_recon, P_actual).mean(dim=[1,2,3])
        loss_clean = self.l1(P_recon, torch.zeros_like(P_recon)).mean(dim=[1,2,3])

        alpha = min(epoch / self.tau, 1.0)
        # calculate combined loss for clean and pois
        loss = ((1 - is_clean) * loss_poison + is_clean * self.psi * loss_clean)
        loss = (1 - alpha) * loss_poison + alpha * loss
        return loss.mean()