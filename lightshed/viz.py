import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from metric import compute_entropy_from_data, ssim_score, compute_metrics
# from nightshade import clip_test


def viz_performance(data):
    '''shows model performance (true vs pred labels) on input image
        compared to true vs reconstructed poison signal'''

    orig_imgs = data['orig_img']
    input_imgs = data["input_img"]
    pred_labels = data["is_poisoned"] # 1 = poisoned, 0 = clean
    true_labels = data['true_labels']
    p_recon = data['p_prime']

    n = 8
    plt.figure(figsize=(12, 3 * n))
    for i in range(n):
        idx = i 

        orig = orig_imgs[idx] # original clean image
        # print(orig.shape)
        poison = p_recon[idx] # poison reconstruction 
        pred_label = pred_labels[idx] 
        true_label = true_labels[idx]
        inp = input_imgs[idx] # input (clean or poisoned)
        true_poison = inp - orig

        # Col 1: input
        plt.subplot(n, 3, i * 3 + 1)
        plt.title(f"True Label ({int(true_label)}) Pred Label({int(pred_label)})")
        plt.imshow(to_img(inp))
        plt.axis("off")

        # Col 2: true poison
        plt.subplot(n, 3, i * 3 + 2)
        plt.title("True Poison")
        plt.imshow(normalize(to_img(true_poison)))
        plt.axis("off")

        # Col 3: poison
        plt.subplot(n, 3, i * 3 + 3)
        plt.title("Poison reconstruction")
        plt.imshow(normalize(to_img(poison)))
        plt.axis("off")
        plt.tight_layout()

    plt.savefig('results/poison_recon_performance.png')
    # plt.show()


def lightshed_process(data):
    orig_imgs = data['orig_img']
    input_imgs = data["input_img"]
    clean_imgs = data['clean_images']
    p_recon = data['p_prime']
    true_label = data['true_labels']
    pred_labels = data["is_poisoned"]

    n = 8
    plt.figure(figsize=(12, 3 * n))

    for i in range(n):
        idx = i

        orig = orig_imgs[idx]
        inp = input_imgs[idx]
        recon_poison = p_recon[idx]
        clean = clean_imgs[idx]
        label = true_label[idx]
        pred_label = pred_labels[idx]
        true_poison = inp - orig

        t_label = 'poison'if label == 1 else "clean"
        p_label = 'poison' if pred_label == 1 else "clean"


        # Col 1: poisoned input
        plt.subplot(n, 4, i * 4 + 1)
        plt.title(f"Input (True Label {int(label)}, {t_label})")
        plt.imshow(to_img(inp))
        plt.axis("off")

        # Col 2: true poison
        plt.subplot(n, 4, i * 4 + 2)
        plt.title("True Poison (Input - Original)")
        plt.imshow(normalize(to_img(true_poison)))
        plt.axis("off")

        # Col 3: reconstructed poison
        plt.subplot(n, 4, i * 4 + 3)
        plt.title(f"Recon. Poison (Pred Label {int(pred_label)}, {p_label})")
        plt.imshow(normalize(to_img(recon_poison)))
        plt.axis("off")
        plt.tight_layout()

        # Col 4: depoisoned image
        plt.subplot(n, 4, i * 4 + 4)
        plt.title("Depoisoned Image")
        plt.imshow(to_img(clean))
        plt.axis("off")
        
    plt.savefig('results/lightshed_process.png')
    # plt.show()

def plot_roc(data, ent_recon):
    true_labels = data['true_labels']
    fpr, tpr, thresh = roc_curve(true_labels, ent_recon)
    auc = roc_auc_score(true_labels, ent_recon)
    

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    # random baseline
    plt.plot([0, 1], [0, 1], linestyle='--')  
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (LightShed)")
    plt.legend()
    plt.grid()
    plt.savefig("results/roc_curve.png")  

def entr_dist(data, entropies):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    true_labels = data['true_labels']
    clean_color = "steelblue"
    poison_color = "darkorange"
    
    poison_mask = true_labels == 1
    clean_mask = true_labels == 0

    num_bins = 12
    bins = np.linspace(min(entropies), max(entropies), num_bins)
    print('lenc',len(entropies[clean_mask]))
    print('lenp', len(entropies[poison_mask]))

    ax.hist(
        entropies[clean_mask],
        bins=bins,
        color=clean_color,
        alpha=0.5,
        #density=True,
        label="Clean"
    )

    ax.hist(
        entropies[poison_mask],
        bins=bins,
        color=poison_color,
        alpha=0.5,
        #density=True,
        label="Poisoned"
    )
    tau_ent = 0.924101
    ax.axvline(tau_ent, color="black", linestyle="--", linewidth=1, label=f"Cutoff (τ = {tau_ent:.3f})")

    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Distribution")

    ax.legend(frameon=True)
    plt.savefig('results/entropy_distribution.png')
    # plt.show()

def ent_vs_ssim(data, ssim_scores, entropies):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    true_labels = data['true_labels']
    clean_color = "steelblue"
    poison_color = "darkorange"
    
    poison_mask = true_labels == 1
    clean_mask = true_labels == 0

    ax.scatter(
        ssim_scores[clean_mask],
        entropies[clean_mask],
        c=clean_color,
        label="Clean",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3
    )

    ax.scatter(
        ssim_scores[poison_mask],
        entropies[poison_mask],
        c=poison_color,
        label="Poisoned",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3
    )

    tau_ent = 0.924101
    ax.axhline(tau_ent, linestyle="--", color="gray", label=r"$\tau_{ENT}$")

    ax.set_xlabel("SSIM (reconstruction vs reference)")
    ax.set_ylabel("Entropy (reconstruction)")
    ax.set_title("A. SSIM–Entropy Decision Plane")

    ax.legend(loc="best", frameon=True)
    plt.savefig('results/ent_vs_ssim.png')
    # plt.show()

def ssim_dist(data, ssim_scores):
    true_labels = data['true_labels']
    clean_color = "steelblue"
    poison_color = "darkorange"
    
    poison_mask = true_labels == 1
    clean_mask = true_labels == 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    num_bins = 12
    bins = np.linspace(min(ssim_scores), max(ssim_scores), num_bins)

    ax.hist(
        ssim_scores[clean_mask],
        bins=bins,
        color=clean_color,
        alpha=0.5,
        density=True,
        label="Clean"
    )

    ax.hist(
        ssim_scores[poison_mask],
        bins=bins,
        color=poison_color,
        alpha=0.5,
        density=True,
        label="Poisoned"
    )

    ax.set_xlabel("SSIM")
    ax.set_ylabel("Density")
    ax.set_title("B. SSIM Distribution")

    ax.legend(frameon=True)
    plt.savefig('results/ssim_distribution.png')
    # plt.show()

def accuracy_table(df):
    fig, ax = plt.subplots()
    ax.axis('off')

    ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center'
    )
    plt.savefig("results/table.png")
    # plt.show()

    
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
    ent_true, ent_recon = compute_entropy_from_data(data)
    ssim_scores = ssim_score(data)
    metric_df = compute_metrics(data)
    lightshed_process(data)
    viz_performance(data)
    accuracy_table(metric_df)
    plot_roc(data, ent_recon)
    entr_dist(data, ent_recon)
    ssim_dist(data, ssim_scores)
    ent_vs_ssim(data, ssim_scores, ent_recon)
