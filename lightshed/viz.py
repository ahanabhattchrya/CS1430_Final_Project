import matplotlib.pyplot as plt
import numpy as np
import torch



def create_comp(data):
    clean_imgs = data['clean_images']
    orig_imgs = data['orig_img']
    input_imgs = data["input_img"]
    pred_labels = data["is_poisoned"] # 1 is poisoned
    true_labels = data['true_labels']
    p_recon = data['p_prime']

    n = 8
    plt.figure(figsize=(12, 3 * n))
    for i in range(n):
        idx = i  # change this to view different samples

        orig = orig_imgs[idx] # original clean image
        # print(orig.shape)
        depoisoned = clean_imgs[idx] # depoisoned images
        poison = p_recon[idx] # poison reconstruction 
        pred_label = pred_labels[idx] 
        true_label = true_labels[idx]
        inp = input_imgs[idx] # input (clean or poisoned)


        # Col 1: original clean
        plt.subplot(n, 4, i * 4 + 1)
        plt.title("Orig Clean")
        plt.imshow(to_img(orig))
        plt.axis("off")

        # Col 2: input
        plt.subplot(n, 4, i * 4 + 2)
        plt.title(f"True Label ({int(true_label)}) Pred Label({int(pred_label)})")
        plt.imshow(to_img(inp))
        plt.axis("off")

        # Col 3: depoisoned
        plt.subplot(n, 4, i * 4 + 3)
        plt.title("Depoisoned")
        plt.imshow(to_img(depoisoned))
        plt.axis("off")

        # Col 4: poison
        plt.subplot(n, 4, i * 4 + 4)
        plt.title("Poison reconstruction")
        plt.imshow(normalize(to_img(poison)))
        plt.axis("off")
        plt.tight_layout()
    plt.savefig(f'inf_plot.png')

def to_img(x):
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return x

def normalize(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


if __name__ == "__main__":
    data = np.load("inference_outputs.npz")
    
    create_comp(data)