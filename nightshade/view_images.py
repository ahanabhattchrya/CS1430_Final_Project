import os
import glob
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button


class PairViewer:
    def __init__(self, input_dir, output_dir, pairs_per_page=10):
        self.pairs_per_page = pairs_per_page
        self.page = 0

        self.input_dir = input_dir
        self.output_dir = output_dir

        self.input_files = glob.glob(os.path.join(input_dir, "*.p"))
        self.output_files = glob.glob(os.path.join(output_dir, "*.p"))

        with open(os.path.join(output_dir, "index_mapping.json"), "r") as f:
            self.map_out_to_in = {int(k): int(v) for k, v in json.load(f).items()}

        self.in_data = self.load_data(self.input_files)
        self.out_data = self.load_data(self.output_files)

        self.paired_indices = [
            (out_idx, in_idx)
            for out_idx, in_idx in self.map_out_to_in.items()
            if out_idx in self.out_data and in_idx in self.in_data
        ]

        self.total_pages = (len(self.paired_indices) + self.pairs_per_page - 1) // self.pairs_per_page

        self.fig, self.axes = plt.subplots(2, self.pairs_per_page, figsize=(18, 4))
        self.fig.subplots_adjust(bottom=0.15, top=0.88)

        ax_prev = plt.axes([0.35, 0.02, 0.15, 0.07])
        ax_next = plt.axes([0.55, 0.02, 0.15, 0.07])

        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")

        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)

        self.render()

    def load_data(self, files):
        data = {}
        for f in files:
            idx = int(os.path.basename(f).split(".")[0])
            with open(f, "rb") as fp:
                obj = pickle.load(fp)
            data[idx] = obj["img"]
        return data

    def render(self):
        for ax in self.axes.flatten():
            ax.clear()
            ax.axis("off")

        start = self.page * self.pairs_per_page
        end = min(start + self.pairs_per_page, len(self.paired_indices))

        for col, (out_idx, in_idx) in enumerate(self.paired_indices[start:end]):
            in_img = Image.fromarray(self.in_data[in_idx])
            out_img = Image.fromarray(self.out_data[out_idx])

            self.axes[0, col].imshow(in_img)
            self.axes[0, col].set_title(f"{in_idx}", fontsize=7)

            self.axes[1, col].imshow(out_img)
            self.axes[1, col].set_title(f"{out_idx}", fontsize=7)

        self.axes[0, 0].set_ylabel("Input Images", fontsize=12)
        self.axes[1, 0].set_ylabel("Poisoned Output Images", fontsize=12)

        self.fig.text(0.1, 0.72, "Input", rotation=90,
                  fontsize=12, va="center", ha="center")

        self.fig.text(0.1, 0.28, "Poisoned Output", rotation=90,
                    fontsize=12, va="center", ha="center")

        self.fig.suptitle(
            f"Clean Input - Filtered Output Pairs | Page {self.page + 1}/{self.total_pages}",
            fontsize=13
        )

        self.fig.canvas.draw_idle()

    def next_page(self, event):
        if self.page < self.total_pages - 1:
            self.page += 1
            self.render()

    def prev_page(self, event):
        if self.page > 0:
            self.page -= 1
            self.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    viewer = PairViewer(args.input_dir, args.output_dir)
    plt.show()
