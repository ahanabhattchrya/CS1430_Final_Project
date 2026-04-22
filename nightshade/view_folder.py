import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image_from_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    img = obj["img"]

    if isinstance(img, Image.Image):
        img = np.array(img)

    return img


def show_folder(folder, cols=5):
    files = sorted(glob.glob(os.path.join(folder, "*.p")))

    if len(files) == 0:
        raise ValueError("No .p files found in folder")

    images = [(os.path.basename(f), load_image_from_pickle(f)) for f in files]

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, (name, img) in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(name, fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)

    args = parser.parse_args()

    show_folder(args.folder)
