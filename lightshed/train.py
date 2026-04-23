# i think algorithm 2 should live here?
# implments models and lightshed_loss 


from models import Encoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightshed_loss import LightShedLoss

import hyperparameters as hp

def train(train_loader, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Encoder.to(device)
    loss_fn = LightShedLoss()

    loss_val = 0

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

def comp_entropy(i):
    B = i.size(0)           # B is batch size
    i = i.view(B, -1)       # flatten the image

    x_min = x.min(dim=1, keepdim=True)[0] #normalize
    x_max = x.max(dim=1, keepdim=True)[0]
    x = (x - x_min) / (x_max - x_min + 1e-8)

    entropies = []
    for i in range(B):
        hist = torch.histc(x[i], bins=256, min=0, max=1)
        p = hist / torch.sum(hist)
        ent = -torch.sum(p * torch.log(p + 1e-8))
        entropies.append(ent)

    return torch.stack(entropies)
    
def comp_threshold(model, val_loader):
    model.eval()

    entropy_list = []
    labels_list = []

    with torch.no_grad():
        for I, I_cor, labels in val_loader:
            I = I.to(DEVICE)

            P_hat = model(I)
            ent = compute_entropy_batch(P_hat)

            all_entropy.extend(ent.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    entropy_list = np.array(entropy_list)
    labels_list = np.array(labels_list)

    fpr, tpr, thresholds = roc_curve(labels_list, entropy_list)

    best_T = np.argmax(tpr - fpr)
    T = thresholds[best_T]

    return T






    

    

    












    # model = LightShedAE from models.py
    # loss_fn = LightShedLoss from lightshed_loss.py
    
