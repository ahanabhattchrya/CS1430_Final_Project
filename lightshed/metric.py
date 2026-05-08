from sklearn.metrics import accuracy_score
import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import torch
import pandas as pd

def compute_metrics(data):
    pred_labels = data["is_poisoned"] # 1 is poisoned
    true_labels = data['true_labels']


    acc = accuracy_score(true_labels, pred_labels)

    TP = ((pred_labels == 1) & (true_labels == 1)).sum() # identify poisoned images
    TN = ((pred_labels == 0) &( true_labels == 0)).sum()
    FP = ((pred_labels == 1) & (true_labels == 0)).sum()
    FN = ((pred_labels == 0 ) & (true_labels == 1)).sum()

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP+TN)
    print(f'Test Accuracy: {acc}, TPR: {TPR}, TNR: {TNR}')
    
    metric_dict = {
    "Poison": ["Nightshade"], 
    "Accuracy": [acc], 
    "TPR": [TPR], 
    "TNR": [TNR]}

    df = pd.DataFrame(metric_dict).round(3)
    
    return df

def ssim_score(data):
    '''real vs reconstructed poison
    average ssim across all samples'''
    orig_imgs = data['orig_img']
    input_imgs = data['input_img']
    p_recon = data['p_prime']
    true_poison = input_imgs - orig_imgs

    scores = []

    for i in range(len(true_poison)):
        poison = true_poison[i]
        # print(poison.shape)
        recon = p_recon[i]
        # print(recon.shape)

        poison = np.transpose(poison, (1, 2, 0))
        recon = np.transpose(recon, (1, 2, 0))
        img_true = img_as_float(poison)
        img_recon = img_as_float(recon)
        # print(img_recon)
        # print(img_true.shape)

        score = ssim(
            img_true,
            img_recon,
            channel_axis=-1, 
            data_range=1.0
        )
        scores.append(score)

    mean_score = np.mean(scores)
    print(f"Average SSIM: {mean_score}")
    return np.array(scores)

def comp_entropy(x):
    # Shannon entropy: information theory measures randomness
        # H(X) = - \sum^n_{i=1} P(x_i) log_2 P(x_i)
    eps = 1e-8
    # normalize per-sample BEFORE softmax
    x_norm = (x - x.mean(dim=(1,2,3), keepdim=True)) / (x.std(dim=(1,2,3), keepdim=True) + eps)

    p_x = torch.softmax(x_norm, dim=1)
    H = - (p_x * torch.log(p_x + eps)).sum(dim=1)
    return H.mean(dim=(1, 2))

def compute_entropy_from_data(data):
    orig = torch.tensor(data['orig_img'], dtype=torch.float32)
    inp = torch.tensor(data['input_img'], dtype=torch.float32)
    recon = torch.tensor(data['p_prime'], dtype=torch.float32)

    true_poison = inp - orig

    ent_true = comp_entropy(true_poison)
    # print(len(ent_true))
    ent_recon = comp_entropy(recon)
    # print(len(ent_recon))
    return ent_true.detach().cpu().numpy(), ent_recon.detach().cpu().numpy()

