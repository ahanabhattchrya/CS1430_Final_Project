import torch
from sklearn.metrics import accuracy_score
import numpy as np

# algorithm 1 from paper in section 4.3 Poison Reconstruction 
def perform_inference(model, T, test_loader, device, save_file='inference_outputs.npz'):

    '''detection and depoisoning
    
    Returns: 
        clean_images: [b, c, h, w]
        is_poisoned: [b]
        p_prime: [b, c, h, w]'''
    model.eval()
    all_preds = []
    all_labels = []

    all_clean_imgs = []
    all_p_prime = []

    orig_img = []
    input_img = []
    with torch.no_grad():
        for I, I_cor, is_clean in test_loader:
            
            # is_clean: 1 = clean, 0 = poisoned
            # I (clear or poisoned), I_cor (clean_image)
            orig_img.append(I_cor)
            input_img.append(I)

            # 1. reconstruction
            I = I.to(device)
            I_cor = I_cor.to(device)
            P_actual = I - I_cor
            is_clean = is_clean.to(device)
            p_prime = model(I) # [b, c, h, w]

            # 2. entropy cut-off
            H = comp_entropy(p_prime)
            is_poisoned = (H > T).float() # boolean [b], 1 = poisoned, 0 = clean 

            # 3. poison subtraction to only inputs predicted as poison
            clean_images = I.clone()
            poison_mask = is_poisoned.bool()
            clean_images[poison_mask] = (I[poison_mask] - p_prime[poison_mask]) # [b, c, h, w]

            all_preds.append(is_poisoned) 
            all_labels.append((1 - is_clean)) # NOW: 1 = poisoned, 0 = clean

            all_clean_imgs.append(clean_images)
            all_p_prime.append(p_prime)

    orig_img = torch.cat(orig_img)
    input_img = torch.cat(input_img)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_clean_imgs = torch.cat(all_clean_imgs)
    all_p_prime = torch.cat(all_p_prime)

    acc = accuracy_score(all_labels, all_preds)

    TP = ((all_preds == 1) & (all_labels == 1)).sum() # identify poisoned images
    TN = ((all_preds == 0) &( all_labels == 0)).sum()
    FP = ((all_preds == 1) & (all_labels == 0)).sum()
    FN = ((all_preds == 0 ) & (all_labels == 1)).sum()

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    print(f'accuracy: {acc}, tpr: {TPR}, tnr: {TNR}')
    np.savez(save_file,
    clean_images=all_clean_imgs.cpu().numpy(),
    orig_img=orig_img.cpu().numpy(),
    input_img=input_img.cpu().numpy(),
    is_poisoned=all_preds,
    true_labels=all_labels,
    p_prime=all_p_prime.cpu().numpy())
    # print(f'tpr: {TPR}')
    # print(f'tnr: {TNR}')
    return clean_images, orig_img, is_poisoned, p_prime

def comp_entropy(x):
    # Shannon entropy: information theory measures randomness
        # H(X) = - \sum^n_{i=1} P(x_i) log_2 P(x_i)
    eps = 1e-8
    # normalize per-sample BEFORE softmax
    x_norm = (x - x.mean(dim=(1,2,3), keepdim=True)) / (x.std(dim=(1,2,3), keepdim=True) + eps)
    p_x = torch.softmax(x_norm, dim=1)
    H = - (p_x * torch.log(p_x + eps)).sum(dim=1)
    
    return H.mean(dim=(1, 2))

