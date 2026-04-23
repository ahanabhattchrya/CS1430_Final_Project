# i think algorithm 2 should live here?
# implments models and lightshed_loss 


from models import Encoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightshed_loss import LightShedLoss

import hyperparameters as hp

def train(device = "cpu", train_loader, val_loader):
    model = Encoder.to(device)
    loss_fn = LightShedLoss()

    loss_val = 0

    # I think here is where the training dataset will go?
    for epoch in range(EPOCHS):

        model.train()

        for I, I_cor, is_clean in train_loader:

            I = I.to(device)    #np and p images
            I_cor = I_cor.to(device)    #from D_cor 
            is_clean = is_clean.to(device) 

            optimizer.zero_grad()

            P_hat = model(I)
            P_actual = I - I_cor

            loss = loss_fn(P_hat, P_actual, is_clean, epoch)

            loss.backward()
            optimizer.step()

            loss_val += loss.item()
        
        avg_loss = loss_val / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    T = comp_thresh(model, val_loader)
    print(f"Threshold T = {T.6f}")

    return model, T

def comp_entropy():
    pass
    
def comp_threshold(model, val_loader):
    pass



    

    

    












    # model = LightShedAE from models.py
    # loss_fn = LightShedLoss from lightshed_loss.py
    
