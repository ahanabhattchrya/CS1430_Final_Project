# CS1430_Final_Project

This is the README.md for the final project

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

oscar for nightshade poisoning
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

## References
Nightshade code base: https://github.com/Shawn-Shan/nightshade-release
