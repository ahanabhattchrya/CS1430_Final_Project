# i think algorithm 2 should live here?
# implments models and lightshed_loss 
from models import Encoder, LightShedAE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightshed_loss import LightShedLoss
from torch.utils.data import DataLoader, random_split
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image
import torchvision.transforms as T
from inference import perform_inference
import matplotlib.pyplot as plt

import hyperparameters as hp

def train(train_loader, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightShedAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.LR)
    loss_fn = LightShedLoss()

    for epoch in range(hp.EPOCHS):

        model.train()
        loss_val = 0

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
    
    T = comp_threshold(model, val_loader, device)
    print(f"Threshold T = {T:.6f}")

    return model, T

def comp_entropy(x):
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
    
def comp_threshold(model, val_loader, device):
    model.eval()

    entropy_list = []
    labels_list = []

    with torch.no_grad():
        for I, I_cor, labels in val_loader:
            # clean = 1, poisoned = 0
            I = I.to(device)

            P_hat = model(I)
            ent = comp_entropy(P_hat)
            # tensors to np and add
            entropy_list += ent.detach().cpu().tolist()
            labels_list += labels.detach().cpu().tolist()
    
    entropy_list = np.array(entropy_list)
    labels_list = np.array(labels_list)
    
    
    plabels_list = 1 - labels_list
    print("poison entropy mean:", entropy_list[plabels_list == 1].mean())
    print("clean entropy mean:", entropy_list[plabels_list == 0].mean())
    auc = roc_auc_score(plabels_list, entropy_list)
    print(f'auc: {auc}')

    fpr, tpr, thresholds = roc_curve(plabels_list, entropy_list)

    #visualize roc

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')  # random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (LightShed)")
    plt.legend()
    plt.grid()
    plt.savefig("roc_curve.png")   # saves to file
    plt.close()
    
    # valid = np.where(fpr >= 0.1)[0]
    # best_T = thresholds[valid[np.argmax(tpr[valid])]]
    best_T = np.argmax(tpr - fpr)
    T_ = thresholds[best_T]
    
    return T_

# To detect poison
def detect(model, dataloader, T, device):
    model.eval()
    pred_vals = []

    with torch.no_grad():
        for I, _, _ in dataloader:
            I = I.to(device)

            P_hat = model(I)
            ent = comp_entropy(P_hat)

            pred = (ent > T).int()
            pred_vals.extend(pred.cpu().numpy())

    return np.array(pred_vals)

# create the dataset to train lightshed with
class LightShedDataset(Dataset):
    def __init__(self, clean_dir, poisoned_dir):
        self.clean_dir = clean_dir
        self.poisoned_dir = poisoned_dir

        self.clean_files = sorted(os.listdir(clean_dir))
        self.poisoned_files = sorted(os.listdir(poisoned_dir))
        self.transform = T.Compose([T.Resize((256, 256)),
                                    T.ToTensor()])

        self.data = []

        for f in self.clean_files:
            self.data.append(("clean", f))

        for f in self.poisoned_files:
            self.data.append(("poisoned", f))

    # def load_p(self, path):
    #     with open(path, "rb") as f:
    #         img = pickle.load(f)

    #     img = torch.tensor(img, dtype=torch.float32)

    #     if img.ndim == 2:
    #         img = img.unsqueeze(0)

    #     return img

    def load_p(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # print(data)
        img = data["img"] 
        # print(type(img))
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        # img = self.transform(img)   
        # img = Image.fromarray(img.astype(np.uint8))

        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_type, fname = self.data[idx]

        if label_type == "clean":
            I = self.load_p(os.path.join(self.clean_dir, fname))
            I_cor = I.copy()
            is_clean = 1
        else:
            I = self.load_p(os.path.join(self.poisoned_dir, fname))
            I_cor = self.load_p(os.path.join(self.clean_dir, fname))
            is_clean = 0
        I = self.transform(I)
        I_cor = self.transform(I_cor)
        return I, I_cor, torch.tensor(is_clean, dtype=torch.float32)



if __name__ == "__main__":
    
    clean_dir = "data/clean"
    pois_dir = "data/poisoned"

    dataset = LightShedDataset(clean_dir, pois_dir)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=hp.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hp.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=hp.BATCH_SIZE, shuffle=False)

    model, T_ = train(train_loader, val_loader)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_vals = detect(model, test_loader, T, device)

    clean_images, orig_img, is_poisoned, p_prime = perform_inference(model, T_, test_loader, device)

