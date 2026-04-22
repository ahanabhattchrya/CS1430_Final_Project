import os
import json
import sys
from PIL import Image
import glob
import argparse
import pickle
from torchvision import transforms
from opt import PoisonGeneration
import torch


def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


def main():
    mapping_dir = os.path.abspath(args.mapping_dir if args.mapping_dir else "mapping_data")
    os.makedirs(mapping_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    poison_generator = PoisonGeneration(target_concept=args.target_name, device=device, eps=args.eps)
    # all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    all_data_paths = sorted(
        all_data_paths,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    input_index_map = {
        idx: int(os.path.splitext(os.path.basename(path))[0])
        for idx, path in enumerate(all_data_paths)
    }
    all_imgs = [pickle.load(open(f, "rb"))['img'] for f in all_data_paths]
    all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    all_imgs = [Image.fromarray(img) for img in all_imgs]

    all_result_imgs = poison_generator.generate_all(all_data_paths, args.target_name)
    os.makedirs(args.outdir, exist_ok=True)

    # for idx, cur_img in enumerate(all_result_imgs):
    #     cur_data = {"text": all_texts[idx], "img": cur_img}
    #     pickle.dump(cur_data, open(os.path.join(args.outdir, "{}.p".format(idx)), "wb"))
    mapping = {}

    for out_idx, cur_img in enumerate(all_result_imgs):
        in_path = all_data_paths[out_idx]
        in_idx = int(os.path.splitext(os.path.basename(in_path))[0])

        cur_data = {
            "text": all_texts[out_idx],
            "img": cur_img
        }

        pickle.dump(
            cur_data,
            open(os.path.join(args.outdir, f"{out_idx}.p"), "wb")
        )
        mapping[out_idx] = in_idx

    with open(os.path.join(mapping_dir, "index_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str,
                        help="", default='')
    parser.add_argument('-od', '--outdir', type=str,
                        help="", default='poisoned_outputs')
    parser.add_argument('-e', '--eps', type=float, default=0.04)
    parser.add_argument('-t', '--target_name', type=str, default="cat")
    parser.add_argument('-md', '--mapping_dir', type=str, default='poison_map')
    return parser.parse_args(argv)

if __name__ == '__main__':
    import time

    args = parse_arguments(sys.argv[1:])
    main()