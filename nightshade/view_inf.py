import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def to_pil(img):
    img = np.array(img)
    img = np.squeeze(img)

    if img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[-1] not in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)


def load_and_save_and_view(npz_path, out_dir="filtered_outputs", max_show=10):
    data = np.load(npz_path, allow_pickle=True)

    print("Keys in .npz file:", data.files)

    label_mask = (data["true_labels"] == 1) & (data["is_poisoned"] == True)
    idxs = np.where(label_mask)[0]
    
    keys = ["clean_images", "orig_img", "input_img"]
    new_keys = ["depoisoned", "clean", "poisoned"]

    key_map = dict(zip(keys, new_keys))

    os.makedirs(out_dir, exist_ok=True)

    print(f"\nTotal matching samples: {len(idxs)}")

    for n, i in enumerate(idxs):
        sample_dir = os.path.join(out_dir, f"{n:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        imgs = []

        for old_key, new_key in zip(keys, new_keys):
            img = to_pil(data[old_key][i])
            imgs.append(img)

            img.save(os.path.join(sample_dir, f"{new_key}.png"))

        if n < max_show:
            plt.figure(figsize=(10, 3))

            for j in range(3):
                plt.subplot(1, 3, j + 1)
                plt.imshow(imgs[j])
                plt.title(new_keys[j])
                plt.axis("off")

            plt.suptitle(f"Sample {n} (true_label=1, poisoned=1)")
            plt.tight_layout()
            plt.show()

    print(f"\nSaved {len(idxs)} samples to {out_dir}/")


if __name__ == "__main__":
    load_and_save_and_view("inference_outputs.npz")
