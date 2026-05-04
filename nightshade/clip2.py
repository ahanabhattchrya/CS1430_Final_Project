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


# ----------------------------
# DUMBBELL PLOT (3 POINTS)
# ----------------------------
def plot_dumbbells(clean, depoison, poison, max_samples=50, save_path=None):
    n = min(len(clean), max_samples)
    idx = np.arange(n)

    plt.figure(figsize=(10, 6))

    for i in range(n):
        # lines connecting all 3 states
        plt.plot(
            [clean[i], depoison[i], poison[i]],
            [i, i, i],
            color="gray",
            alpha=0.5,
            linewidth=2
        )

    # scatter points
    plt.scatter(clean[:n], idx, label="Clean", color="#4ee040", s=40)
    plt.scatter(depoison[:n], idx, label="Depoisoned", color="#f0aa53", s=40)
    plt.scatter(poison[:n], idx, label="Poisoned", color="#9b0d0d", s=40)

    # # mean reference lines
    # plt.axvline(clean.mean(), color="#1f77b4", linestyle="--")
    # plt.axvline(depoison.mean(), color="#2ca02c", linestyle="--")
    # plt.axvline(poison.mean(), color="#ff7f0e", linestyle="--")
    plt.yticks(idx, ["0", "1", "2"][:n])
    plt.xlabel("CLIP similarity to target ('a photo of a cat')")
    plt.ylabel("Image index")
    plt.title("Per-sample CLIP Shift: Clean → Depoisoned → Poisoned")

    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()
    plt.show()

    print("\n=== Means ===")
    print("Clean:", clean.mean())
    print("Depoisoned:", depoison.mean())
    print("Poisoned:", poison.mean())


# ----------------------------
# MAIN
# ----------------------------
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

    plot_dumbbells(clean, depoison, poison, save_path="results/clip2.png")


if __name__ == "__main__":
    run("filtered_outputs")
