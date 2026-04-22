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
pip install scikit-image
```

## Running Code
### Nightshade
data_extraction.py
```
python data_extraction.py --directory data/ --concept dog --num 100
```

view_images.py 

(view images with an index_mapping json between input-output)
```
python view_images.py --mapping_dir mapping_data --input_dir data --output_dir selected_data --has_json true
```
(view poisoned image by matching input/output file names (0-100.p))

```
python view_images.py --input_dir data --output_dir poisoned_outputs
```

gen_poison.py
```
python gen_poison.py --directory selected_data --target_name cat
```
oscar for nightshade poisoning
```
sbatch slurm_nightshade.sh
```

## References
Nightshade code base: https://github.com/Shawn-Shan/nightshade-release
