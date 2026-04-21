# CS1430_Final_Project

This is the README.md for the final project

## Installation
```
#!/bin/bash

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
pip install numpy pillow matplotlib
pip install torch torchvision
pip install scikit-learn
pip install git+https://github.com/openai/CLIP.git
```