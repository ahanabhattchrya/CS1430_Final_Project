# i think algorithm 2 should live here?
# implments models and lightshed_loss 


from models import Encoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightshed_loss import LightShedLoss


def train(device = "cpu"):
    model = Encoder.to(device)
    loss_fn = LightShedLoss()





    # model = LightShedAE from models.py
    # loss_fn = LightShedLoss from lightshed_loss.py
    
