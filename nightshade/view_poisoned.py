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


def show_paired_folders(input_dir, output_dir, cols=10):
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.p")))
    output_files = sorted(glob.glob(os.path.join(output_dir, "*.p")))

    if len(input_files) == 0 or len(output_files) == 0:
        raise ValueError("No .p files found in one of the folders")

    # build index → file map
    input_map = {
        int(os.path.basename(f).split(".")[0]): f
        for f in input_files
    }

    output_map = {
        int(os.path.basename(f).split(".")[0]): f
        for f in output_files
    }

    common_indices = sorted(set(input_map.keys()) & set(output_map.keys()))

    rows = (len(common_indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols, figsize=(2.5 * cols, 5 * rows))
    axes = np.array(axes)

    # turn off all axes
    for ax in axes.flatten():
        ax.axis("off")

    for i, idx in enumerate(common_indices):
        r = (i // cols) * 2
        c = i % cols

        in_img = load_image_from_pickle(input_map[idx])
        out_img = load_image_from_pickle(output_map[idx])

        axes[r, c].imshow(in_img)
        axes[r, c].set_title(f"in {idx}", fontsize=7)

        axes[r + 1, c].imshow(out_img)
        axes[r + 1, c].set_title(f"out {idx}", fontsize=7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    show_paired_folders(args.input_dir, args.output_dir)
