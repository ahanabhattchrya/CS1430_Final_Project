import os
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt


def load_img(path):
    return Image.open(path).convert("RGB")


def score_img(img, model, preprocess, text_features, device):
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return (f @ text_features.T).item()


def compute_scores(folder, model, preprocess, text_features, device):
    clean, depoison, poison = [], [], []

    samples = sorted(os.listdir(folder))

    for s in samples:
        path = os.path.join(folder, s)

        c = load_img(os.path.join(path, "clean.png"))
        d = load_img(os.path.join(path, "depoisoned.png"))
        p = load_img(os.path.join(path, "poisoned.png"))

        clean.append(score_img(c, model, preprocess, text_features, device))
        depoison.append(score_img(d, model, preprocess, text_features, device))
        poison.append(score_img(p, model, preprocess, text_features, device))

    return (
        np.array(clean),
        np.array(depoison),
        np.array(poison)
    )

def plot_points(clean, depoison, poison, max_samples=3, save_path=None):
    n = min(len(clean), max_samples)

    clean = clean[:n]
    depoison = depoison[:n]
    poison = poison[:n]

    order = [1, 2, 0][:n]

    clean = clean[order]
    poison = poison[order]
    depoison = depoison[order]

    plt.figure(figsize=(10, 5))

    x = []
    y = []
    labels = []
    colors = []

    for i in range(n):
        idx = order[i]

        x.append(len(x))
        y.append(clean[i])
        labels.append(f"{i}-clean")
        colors.append("#4ee040")

        x.append(len(x))
        y.append(poison[i])
        labels.append(f"{i}-poisoned")
        colors.append("#9b0d0d")

        x.append(len(x))
        y.append(depoison[i])
        labels.append(f"{i}-depoisoned")
        colors.append("#f0aa53")

    plt.scatter(x, y, c=colors, s=80)

    for i in range(n):
        base = i * 3
        plt.plot(
            [base, base + 1, base + 2],
            [y[base], y[base + 1], y[base + 2]],
            color="gray",
            alpha=0.5
        )

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Image (Index-Condition)")
    plt.ylabel("CLIP similarity to target ('a photo of a cat')")
    plt.title("CLIP Similarity per Image (Clean → Poison → Depoisoned)")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()


def run(folder="filtered_outputs", device="cuda"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(["a photo of a cat"]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clean, depoison, poison = compute_scores(
        folder, model, preprocess, text_features, device
    )

    plot_points(clean, depoison, poison, save_path="results/clip2.png")


if __name__ == "__main__":
    run("filtered_outputs")
