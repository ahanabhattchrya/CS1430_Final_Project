import os
import glob
import pickle
import torch
import numpy as np
from PIL import Image
import clip
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wilcoxon


def load_image_from_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    img = obj["img"]

    if isinstance(img, Image.Image):
        return img.convert("RGB")
    else:
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")


def load_paths(folder):
    return sorted(
        glob.glob(os.path.join(folder, "*.p")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )


def plot(clean_scores, poison_scores, save_dir="results", sort=True):

    os.makedirs(save_dir, exist_ok=True)

    clean_scores = np.array(clean_scores)
    poison_scores = np.array(poison_scores)

    delta = poison_scores - clean_scores

    if sort:
        order = np.argsort(delta)
        clean_scores = clean_scores[order]
        poison_scores = poison_scores[order]
        delta = delta[order]

    idx = np.arange(len(clean_scores))

    plt.figure(figsize=(10, 6))
    plt.style.use("default")

    # draw connecting lines
    for i in range(len(idx)):
        color = "#2ca02c" if delta[i] > 0 else "#d62728"
        plt.plot(
            [clean_scores[i], poison_scores[i]],
            [i, i],
            color=color,
            alpha=0.6,
            linewidth=2
        )

    plt.scatter(clean_scores, idx, color="#1f77b4", s=40, label="clean → target", zorder=3)
    plt.scatter(poison_scores, idx, color="#ff7f0e", s=40, label="poison → target", zorder=3)

    plt.axvline(x=np.mean(clean_scores), linestyle="--", color="gray", alpha=0.5)

    plt.xlabel("CLIP similarity of image to target 'cat'", fontsize=12)
    plt.ylabel("Samples (sorted by effect)" if sort else "Samples", fontsize=12)
    plt.title("Effects of Nightshade: Per-image CLIP score shift (poisoned - clean)", fontsize=14)

    legend = plt.legend(
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
    )

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_alpha(0.95)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "clip_dumbbell.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

def plot_histogram(clean_scores, poison_scores, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    clean_scores = np.array(clean_scores)
    poison_scores = np.array(poison_scores)

    bins = np.linspace(
        min(clean_scores.min(), poison_scores.min()),
        max(clean_scores.max(), poison_scores.max()),
        25
    )

    plt.figure(figsize=(8, 5))

    plt.hist(clean_scores, bins=bins, alpha=0.35, color="#1f77b4", label="Clean (counts)")
    plt.hist(poison_scores, bins=bins, alpha=0.35, color="#ff7f0e", label="Poison (counts)")

    # --- means ---
    clean_mean = clean_scores.mean()
    poison_mean = poison_scores.mean()

    plt.axvline(clean_mean, linestyle="--", color="#1f77b4", linewidth=2)
    plt.axvline(poison_mean, linestyle="--", color="#ff7f0e", linewidth=2)

    plt.text(
        clean_mean,
        plt.ylim()[1] * 0.85,
        f"clean mean = {clean_mean:.3f}",
        color="#1f77b4",
        rotation=0,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    plt.text(
        poison_mean,
        plt.ylim()[1] * 0.70,
        f"poison mean = {poison_mean:.3f}",
        color="#ff7f0e",
        rotation=0,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )


    # --- axes ---
    plt.xlabel("CLIP similarity between image and target 'cat")
    plt.ylabel("Image Count (n=100)")
    plt.title("CLIP Score Distribution of Clean and Poisoned Images")

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "clip_histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

def run(clean_folder, poison_folder, target_text, device="cuda"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    clean_paths = load_paths(clean_folder)
    poison_paths = load_paths(poison_folder)

    assert len(clean_paths) == len(poison_paths), "Mismatch in dataset size"

    text_tokens = clip.tokenize([target_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clean_scores = []
    poison_scores = []

    for c_path, p_path in zip(clean_paths, poison_paths):
        clean_img = load_image_from_pickle(c_path)
        poison_img = load_image_from_pickle(p_path)

        c_tensor = preprocess(clean_img).unsqueeze(0).to(device)
        p_tensor = preprocess(poison_img).unsqueeze(0).to(device)

        with torch.no_grad():
            c_feat = model.encode_image(c_tensor)
            p_feat = model.encode_image(p_tensor)

            c_feat = c_feat / c_feat.norm(dim=-1, keepdim=True)
            p_feat = p_feat / p_feat.norm(dim=-1, keepdim=True)

            c_sim = (c_feat @ text_features.T).item()
            p_sim = (p_feat @ text_features.T).item()

        clean_scores.append(c_sim)
        poison_scores.append(p_sim)

    clean_scores = np.array(clean_scores)
    poison_scores = np.array(poison_scores)

    print("\n=== CLIP Evaluation ===")
    print("Avg clean → target:", clean_scores.mean())
    print("Avg poison → target:", poison_scores.mean())
    print("Improvement:", (poison_scores - clean_scores).mean())
    print("Fraction improved:", (poison_scores > clean_scores).mean())

    plot(clean_scores, poison_scores)
    plot_histogram(clean_scores, poison_scores)
    compute_significance(clean_scores, poison_scores)


if __name__ == "__main__":
    run(
        clean_folder="selected_data",
        poison_folder="poisoned_outputs",
        target_text="A photo of a cat",
        device="cuda"
    )
