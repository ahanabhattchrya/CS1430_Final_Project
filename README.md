# CS1430_Final_Project
In this project we are studying Nightshade [(Shan et al. 2024)](https://www.usenix.org/system/files/usenixsecurity25-foerster.pdf), a prompt-specific poisoning attack on text-to-image generative models, and Lightshed [(Foerster et al. 2025)](https://www.usenix.org/system/files/usenixsecurity25-foerster.pdf), a technique that detects and depoisons images that have been Nightshaded. Our goal is to implement the Nightshade algorithm, generate 100 poisoned images of concept dog, target cat, and implement the Lightshed algorithm to detect poisoned images. 

## Installation
```
# Create virtual environment in the folder
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
pip install numpy pillow matplotlib
pip install torch torchvision
pip install scikit-learn
pip install git+https://github.com/openai/CLIP.git
pip install diffusers transformers accelerate safetensors
pip install einops
```

## Running Code
### Nightshade
data_extraction.py
```
python data_extraction.py --directory data/ --concept dog --num 100
```

gen_poison.py
```
python gen_poison.py --directory selected_data --target_name cat
```

oscar for nightshade poisoning (~2 hours for 100 images)
``` 
cd nightshade
source .venv/bin/activate
```
```
sbatch slurm_nightshade.sh
```
view_images.py (view 100 clean v. poisoned images side by side)
```
python view_images.py --input_dir selected_data --output_dir poisoned_outputs
```
visualize histogram for CLIP score comparison between clean-cat and poisoned-cat
```
python clip_test.py # generates results/clip_histogram.png
```

## Lightshed
oscar for Lightshed training (~6 mins)
``` 
cd lightshed
source .venv/bin/activate
```
```
sbatch slurm_lightshed.sh
```

## References
Nightshade code base: https://github.com/Shawn-Shan/nightshade-release
