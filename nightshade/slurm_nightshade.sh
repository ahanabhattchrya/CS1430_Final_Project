#!/bin/bash

# ============================================================
# CSCI1430 - Computer Vision
# Brown University
# SLURM job script
# FINAL PROJECT
#
# Monitor your job:
#   myq                      # check job status
#   cat slurm-<jobid>.out    # view stdout
#   cat slurm-<jobid>.err    # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J hw5_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python gen_poison.py \
    --directory selected_data/ \
    --target_name cat

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"