import torch

# algorithm 1 from paper in section 4.3 Poison Reconstruction i think?
# unsure what to call this file? could just call it inference maybe?
# testing
class Inference:
    def __init__(self, model, T):
        self.model = model # trained autoencoder
        self.T = T # threshold from validation set

    def forward(self, I): # algo1 pipeline
        '''detection and depoisoning
        
        Returns: 
            clean_images: [b, c, h, w]
            is_poisoned: [b]
            p_prime: [b, c, h, w]'''

        # 1. reconstruction
        p_prime = self.model(I) # [b, c, h, w]

        # 2. entropy cut-off
        H = entropy(p_prime)
        is_poisoned = H > self.T # boolean [b]

        # 3. poison subtraction
        clean_images = I.clone()
        clean_images[is_poisoned] = (I[is_poisoned] - p_prime[is_poisoned]) # [b, c, h, w]
        
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