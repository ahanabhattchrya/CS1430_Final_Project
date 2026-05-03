from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torch
import os

def accuracy_scores(data):

    # clean_imgs = data['clean_images']
    # orig_imgs = data['orig_img']
    # input_imgs = data["input_img"]
    pred_labels = data["is_poisoned"] # 1 is poisoned
    true_labels = data['true_labels']
    # p_recon = data['p_prime']

    acc = accuracy_score(true_labels, pred_labels)

    TP = ((pred_labels == 1) & (true_labels == 1)).sum() # identify poisoned images
    TN = ((pred_labels == 0) &( true_labels == 0)).sum()
    FP = ((pred_labels == 1) & (true_labels == 0)).sum()
    FN = ((pred_labels == 0 ) & (true_labels == 1)).sum()

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    print(acc, TPR, TNR)
    return acc, TPR, TNR


def ssim_score(data):
    '''real vs reconstructed poison
    average ssim across all samples'''
    # clean_imgs = data['clean_images']
    orig_imgs = data['orig_img']
    input_imgs = data['input_img']
    p_recon = data['p_prime']
    true_poison = input_imgs - orig_imgs
    print(true_poison.shape)
    print(true_poison.max())
    print(p_recon.shape)

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
            channel_axis=-1,   # use None if grayscale
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
    # B = x.shape[0]
    # x = x.view(B, -1)
    p_x = torch.softmax(x_norm, dim=1)
    H = - (p_x * torch.log(p_x + eps)).sum(dim=1)
    return H.mean(dim=(1, 2))

def compute_entropy_from_data(data):
    orig = torch.tensor(data['orig_img'], dtype=torch.float32)
    inp = torch.tensor(data['input_img'], dtype=torch.float32)
    recon = torch.tensor(data['p_prime'], dtype=torch.float32)

    true_poison = inp - orig

    ent_true = comp_entropy(true_poison)
    ent_recon = comp_entropy(recon)

    return ent_true.detach().cpu().numpy(), ent_recon.detach().cpu().numpy()
    
def plot_metrics(ssim_vals, entropy_diff, pred_labels, true_labels):
    entropy_diff = np.array(entropy_diff)
    ssim_vals = np.array(ssim_vals)
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)

    correct = pred_labels == true_labels

    poison_mask = true_labels == 1
    clean_mask = true_labels == 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    # poison subplot
    plot_subset(
        axes[0],
        entropy_diff[poison_mask],
        ssim_vals[poison_mask],
        correct[poison_mask],
        "Poison Samples"
    )

    # clean subplot
    plot_subset(
        axes[1],
        entropy_diff[clean_mask],
        ssim_vals[clean_mask],
        correct[clean_mask],
        "Clean Samples"
    )

    axes[0].set_ylabel("Value")
    handles = [
    axes[0].scatter([], [], color='blue', label='Entropy Difference'),
    axes[0].scatter([], [], color='red', label='SSIM'),
    axes[0].plot([], [], color='green', linewidth=2, label='Correct Classification')[0],
    axes[0].plot([], [], color='red', linewidth=2, label='Incorrect Classification')[0],
]

    
    plt.suptitle("Entropy Difference vs SSIM (Poison reconstruction vs ground truth)", y=0.98)
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.94))
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    plt.savefig('results/entropy_ssim_comp.png')
    plt.show()

def plot_subset(ax, entropy, ssim, correct, title):
    offset = 0.15
    x = np.arange(len(entropy))
    for i in range(len(x)):
        color = 'green' if correct[i] else 'red'
        lw = 2.5 if not correct[i] else 1.8
        
        ax.plot([x[i], x[i]],
                [entropy[i], ssim[i]],
                color=color, alpha=0.7, linewidth=lw)

    # scatter points
    ax.scatter(x - offset, entropy, color='blue', zorder=3, label='entropy')
    ax.scatter(x + offset, ssim, color='red', zorder=3, label='ssim')

    # labels
    for i in range(len(x)):
        ax.text(x[i] - offset, entropy[i] - 0.05,
                f"{entropy[i]:.2f}", color='blue',
                ha='center', fontsize=7)

        ax.text(x[i] + offset, ssim[i] + 0.02,
                f"{ssim[i]:.2f}", color='red',
                ha='center', fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)

if __name__ == "__main__":
    data = np.load('inference_outputs.npz')
    print(len(data['true_labels']))
    accuracy_scores(data)
    ssim_scores = ssim_score(data)
    n = len(ssim_scores)
    ent_true, ent_recon = compute_entropy_from_data(data)
    entropy_diff = np.abs(ent_true - ent_recon) 

    plot_metrics(ssim_scores, entropy_diff, data['is_poisoned'], data['true_labels'])
    # plot(data)
    # ssim_score(data)