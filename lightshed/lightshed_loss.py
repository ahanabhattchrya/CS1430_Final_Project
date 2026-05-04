
import torch.nn as nn
import torch

class LightShedLoss(nn.Module):
    def __init__(self, psi=0.01, tau=100):
        super().__init__()
        self.psi = psi
        self.tau = tau
        self.l1 = nn.L1Loss(reduction='none')
    
    def ssim_loss(self, P_recon, P_actual):
        # 2 mux muy + c1 * 2sigxy + c2 / mux**2 muy**2+ c1 * sigx**2 sigy**2 + c2
        # p_recon = x, p_actual = y
        mu_x = P_recon.mean(dim=[2, 3], keepdim=True)
        mu_y = P_actual.mean(dim=[2, 3], keepdim=True)

        sigma_x = ((P_recon - mu_x)**2).mean(dim=[2, 3], keepdim=True)
        sigma_y = ((P_actual - mu_y)**2).mean(dim=[2, 3], keepdim=True)
        sigma_xy = ((P_recon - mu_x)*(P_actual - mu_y)).mean(dim=[2,3], keepdim=True)

        C1, C2 = 0.01**2, 0.03**2

        ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

        return 1 - ssim.mean(dim=[1, 2, 3])
    
    def comp_entropy(self, x):
    # Shannon entropy: information theory measures randomness
        # H(X) = - \sum^n_{i=1} P(x_i) log_2 P(x_i)
        eps = 1e-8
        # normalize per-sample BEFORE softmax
        x_norm = (x - x.mean(dim=(1,2,3), keepdim=True)) / (x.std(dim=(1,2,3), keepdim=True) + eps)
        # B = x.shape[0]
        # x = x.view(B, -1)
        p_x = torch.softmax(x_norm, dim=1)
        H = - (p_x * torch.log(p_x + eps)).sum(dim=1)
        return H.mean(dim=(1, 2))
    
    def forward(self, P_recon, P_actual, is_clean, epoch):

        L_ssim = self.ssim_loss(P_recon, P_actual)
        L_recon = torch.abs(P_recon - P_actual).mean(dim=[1, 2, 3])

        err_sq = (P_recon - P_actual) **2
        L_regu = torch.relu(err_sq - self.psi).mean(dim=[1, 2, 3])

        base_loss = 0.7 * L_ssim + 0.2 * L_regu + 0.1 * L_recon
        # loss_poison = self.l1(P_recon, P_actual).mean(dim=[1,2,3])
        # loss_clean = self.l1(P_recon, torch.zeros_like(P_recon)).mean(dim=[1,2,3])

        # alpha = min(epoch / self.tau, 1.0)
        # # calculate combined loss for clean and pois
        # loss = ((1 - is_clean) * loss_poison + is_clean * self.psi * loss_clean)
        # loss = (1 - alpha) * loss_poison + alpha * loss

        if epoch >= self.tau: 
            ent = self.comp_entropy(P_recon)

            # L_ft = (1 - 2 * is_clean) * ent # here clean = 1 --> negative
            L_ft = (2 * is_clean - 1) # clean = --> positive
            loss = base_loss + 0.3 * L_ft

        else: loss = base_loss

        return loss.mean()