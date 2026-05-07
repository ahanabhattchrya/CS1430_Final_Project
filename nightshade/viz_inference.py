import os
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image


def load_npz(path):
    return np.load(path, allow_pickle=True)


def to_clip_tensor(img, preprocess, device):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    return preprocess(img).unsqueeze(0).to(device)

def compute_scores(model, preprocess, images, text_features, device):
    scores = []

    for img in images:
        img_t = to_clip_tensor(img, preprocess, device)

        with torch.no_grad():
            feat = model.encode_image(img_t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            score = (feat @ text_features.T).item()

        scores.append(score)

    return np.array(scores)


def plot_hist_three(a, b, c, save_path):
    plt.figure(figsize=(8, 5))

    all_vals = np.concatenate([a, b, c])
    bins = np.linspace(all_vals.min(), all_vals.max(), 25)

    plt.hist(a, bins=bins, alpha=0.35, color="#1f77b4", label="orig_img → target")
    plt.hist(b, bins=bins, alpha=0.35, color="#ff7f0e", label="poisoned (is_poisoned=1) → target")
    plt.hist(c, bins=bins, alpha=0.35, color="#2ca02c", label="input_img → target")

    plt.xlabel("CLIP similarity to target")
    plt.ylabel("Count")
    plt.title("CLIP Score Comparison Across Image Variants")

    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


def run(npz_path, target_text, device="cuda"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_npz(npz_path)

    model, preprocess = clip.load("ViT-B/32", device=device)

    # text embedding
    text_tokens = clip.tokenize([target_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    poison_mask = data["is_poisoned"].astype(bool)

    orig_imgs = data["orig_img"]
    input_imgs = data["input_img"]
    poison_imgs = data["p_prime"][poison_mask]

    # compute CLIP scores
    orig_scores = compute_scores(model, preprocess, orig_imgs, text_features, device)
    input_scores = compute_scores(model, preprocess, input_imgs, text_features, device)
    poison_scores = compute_scores(model, preprocess, poison_imgs, text_features, device)

    # plot
    os.makedirs("results", exist_ok=True)
    plot_hist_three(
        orig_scores,
        poison_scores,
        input_scores,
        save_path="results/clip_npz_comparison.png"
    )


if __name__ == "__main__":
    run(
        npz_path="inference_outputs.npz",
        target_text="a photo of a cat",
        device="cuda"
    )
