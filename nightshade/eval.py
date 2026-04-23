import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusers import StableDiffusionXLPipeline
from einops import rearrange
from torchvision import transforms
import torch.nn.functional as F

def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512))
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, "h w c -> c h w")
    return torch.tensor(img).unsqueeze(0)

class VAEEncoder:
    def __init__(self, device):
        self.device = device
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)

    def encode(self, img_tensor):
        with torch.no_grad():
            latent = self.pipe.vae.encode(img_tensor.to(self.device).half()).latent_dist.mean
        return latent

def load_images(folder):
    paths = sorted(
        glob.glob(os.path.join(folder, "*.p")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return imgs

def run(clean_folder, poison_folder, target_path, device="cuda"):

    encoder = VAEEncoder(device)

    clean_imgs = load_images(clean_folder)
    poison_imgs = load_images(poison_folder)
    target_img = Image.open(target_path).convert("RGB")

    clean_latents = []
    poison_latents = []

    for c, p in zip(clean_imgs, poison_imgs):
        c_t = img2tensor(c)
        p_t = img2tensor(p)

        clean_latents.append(encoder.encode(c_t))
        poison_latents.append(encoder.encode(p_t))

    clean_latents = torch.cat(clean_latents)
    poison_latents = torch.cat(poison_latents)
    target_latent = encoder.encode(img2tensor(target_img))

    clean_dist = (clean_latents.flatten(1) - target_latent.flatten(1)).norm(dim=1)
    poison_dist = (poison_latents.flatten(1) - target_latent.flatten(1)).norm(dim=1)

    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", "latent_eval.txt")

    avg_clean = clean_dist.mean().item()
    avg_poison = poison_dist.mean().item()
    improvement = (clean_dist - poison_dist).mean().item()
    fraction_improved = (poison_dist < clean_dist).float().mean().item()

    log_text = f"""
    === Latent Evaluation ===
    Avg clean → target: {avg_clean}
    Avg poison → target: {avg_poison}
    Improvement: {improvement}
    Fraction improved: {fraction_improved}
    """

    with open(log_path, "w") as f:
        f.write(log_text)

    print(f"\nSaved latent evaluation to: {log_path}")

    def flat(x): return x.reshape(x.shape[0], -1).cpu().numpy()

    all_data = np.concatenate([
        flat(clean_latents),
        flat(poison_latents),
        flat(target_latent.unsqueeze(0))
    ])

    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_data)

    n = len(clean_imgs)

    plt.figure(figsize=(6,5))
    plt.scatter(proj[:n,0], proj[:n,1], label="clean", alpha=0.6)
    plt.scatter(proj[n:2*n,0], proj[n:2*n,1], label="poison", alpha=0.6)
    plt.scatter(proj[-1,0], proj[-1,1], label="target", marker="*", s=200)

    plt.legend()
    plt.title("Latent Space (PCA)")


    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "latent_pca.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved PCA plot to: {save_path}")


if __name__ == "__main__":
    run(
        clean_folder="selected_inputs",
        poison_folder="poisoned_outputs",
        target_path="target.png",
        device="cuda"
    )
