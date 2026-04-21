import os
import sys

import torch
from PIL import Image
import glob
import pickle
import argparse
from torchvision import transforms
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import clip
import json


def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


class CLIP(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess = clip.load("ViT-B/32", device=self.device)

        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = clip.tokenize

    def text_emb(self, text_ls):
        if isinstance(text_ls, str):
            text_ls = [text_ls]

        text = self.tokenizer(text_ls, truncate=True).to(self.device)

        with torch.no_grad():
            return self.model.encode_text(text)

    def __call__(self, image, text):
        if isinstance(image, list):
            image = [
                self.preprocess(i).unsqueeze(0).to(self.device)
                for i in image
            ]
            image = torch.cat(image)
        else:
            image = self.preprocess(image).unsqueeze(0).to(self.device)

        text = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        return similarity[0][0]


def main():
    clip_model = CLIP()

    data_dir = args.directory
    source_concept = args.concept

    outdir = args.outdir if args.outdir and args.outdir.strip() != "" else "selected_data"
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    all_data = glob.glob(os.path.join(data_dir, "*.p"))
    res_ls = []

    for cur_data_f in all_data:
        idx = int(os.path.basename(cur_data_f).split(".")[0])

        cur_data = pickle.load(open(cur_data_f, "rb"))
        cur_img = Image.fromarray(cur_data["img"])
        cur_text = cur_data["text"]

        cur_img = crop_to_square(cur_img)

        score = clip_model(cur_img, f"a photo of a {source_concept}")

        if score > 0.24:
            res_ls.append((idx, cur_img, cur_text))

    if len(res_ls) < args.num:
        raise ValueError("Not enough data from source concept.")

    all_prompts = [d[2] for d in res_ls]

    text_emb = clip_model.text_emb(all_prompts)
    text_emb_target = clip_model.text_emb(f"a photo of a {source_concept}")

    text_emb_np = text_emb.cpu().float().numpy()
    text_emb_target_np = text_emb_target.cpu().float().numpy()

    res = cosine_similarity(text_emb_np, text_emb_target_np).reshape(-1)

    candidate = np.argsort(res)[::-1][:300]
    random_selected_candidate = random.sample(list(candidate), args.num)

    final_list = [res_ls[i] for i in random_selected_candidate]

    index_mapping = {}

    for i, (orig_idx, img, text) in enumerate(final_list):
        cur_data = {
            "img": np.array(img),
            "text": text,
            "index": int(orig_idx)
        }

        out_path = os.path.join(outdir, f"{i}.p")

        with open(out_path, "wb") as f:
            pickle.dump(cur_data, f)

        index_mapping[i] = int(orig_idx)

    mapping_path = os.path.join(outdir, "index_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(index_mapping, f, indent=2)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--directory', type=str, default='')
    parser.add_argument('-od', '--outdir', type=str, default='')
    parser.add_argument('-n', '--num', type=int, default=100)
    parser.add_argument('-c', '--concept', type=str, required=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
