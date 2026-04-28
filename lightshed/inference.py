import torch
from sklearn.metrics import accuracy_score

# algorithm 1 from paper in section 4.3 Poison Reconstruction i think?
# unsure what to call this file? could just call it inference maybe?
# testing
def perform_inference(model, T, test_loader, device):

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
    with torch.no_grad():
        for I, I_cor, is_clean in test_loader:
            # is_clean 1 = clean, 0 = poisoned
            # I (clear or poisoned), I (clean_image)

            # 1. reconstruction
            I = I.to(device)
            is_clean = is_clean.to(device)
            p_prime = model(I) # [b, c, h, w]

            # 2. entropy cut-off
            H = entropy(p_prime)
            is_poisoned = (H > T).float() # boolean [b], 1 = poisoned, 0 = clean 

            # 3. poison subtraction
            clean_images = I.clone()
            poison_mask = is_poisoned.bool()
            clean_images[poison_mask] = (I[poison_mask] - p_prime[poison_mask]) # [b, c, h, w]

            all_preds.append(is_poisoned) # 
            all_labels.append((1- is_clean)) #

            all_clean_imgs.append(clean_images)
            all_p_prime.append(p_prime)
    
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
    # print(f'tpr: {TPR}')
    # print(f'tnr: {TNR}')
    return clean_images, is_poisoned, p_prime

        ####### non vectorized version (abandoning)#######
        # clean_images = []
        # labels = []
        # recon = []
        # for img in I: # img = [c, h, w]
        # # 1. reconstruction  
        #     # self.model expects [1, c, h, w]
        #     p_prime = self.model(img.unsqueeze(0)) # p_prime = [1, c, h, w]
        #     p_prime = p_prime.squeeze(0) # [c, h, w] for poison subtraction

        # # 2. entropy cut-off + classification (poison or clean)
        #     H_i = entropy(p_prime)
        #     if H_i > self.T:
        #         label = 'poison'
        #         # 3. poison subtraction
        #         img_clean = img - p_prime # [c, h, w]
                
        #     else:
        #         label = 'clean'
        #         img_clean = img
            
        #     clean_images.append(img_clean)
        #     labels.append(label)
        #     recon.append(p_prime) 
                

        # return torch.stack(clean_images), labels, torch.stack(recon)


def entropy(x):
    # x [b, c, h, w]
    # Shannon entropy: information theory measures randomness
        # H(X) = - \sum^n_{i=1} P(x_i) log_2 P(x_i)
    B = x.shape[0]
    x = x.view(B, -1)
    # maybe normalize before softmax ?
    p_x = torch.softmax(x, dim=1)
    H = - (p_x * torch.log2(p_x)).sum(dim=1)
    return H

if __name__ == "__main__":
    
    # clean_dir = "data/clean"
    # pois_dir = "data/poisoned"

    # dataset = LightShedDataset(clean_dir, pois_dir)
    # train_size = int(0.7 * len(dataset))
    # val_size = int(0.15 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    # train_loader = DataLoader(train_set, batch_size=hp.BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=hp.BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=hp.BATCH_SIZE, shuffle=False)

    # model, T = train(train_loader, val_loader)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_vals = detect(model, test_loader, T, device)