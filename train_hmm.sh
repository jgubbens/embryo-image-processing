#!/bin/bash
#SBATCH --job-name=hmm_train
#SBATCH --output=logs/hmm_train_%j.out
#SBATCH --error=logs/hmm_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100   # V100 (CC 7.0) is too old for the current torch wheel; A100 is CC 8.0
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jgubbs64@gmail.com

# --- Environment setup ------------------------------------------------------
# No environment modules needed: uv manages its own Python, and the PyTorch
# wheels bundle their own CUDA runtime (the node's NVIDIA driver is enough).
#
# uv manages the project venv (see pyproject.toml / uv.lock).
# Install it once on the login node with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Run from the repository root so the relative paths in hybrid_hmm_trainer.py
# ('data/training_data', 'models/', 'models/model_info.json') resolve.
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs models

# --- Sync dependencies and run ----------------------------------------------
# --frozen uses the committed uv.lock exactly; drop it if you want resolution.
uv sync --frozen

echo "Node:    $(hostname)"
echo "GPU(s):  $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Unbuffered output so print() lines stream to the logs live (tail -f).
export PYTHONUNBUFFERED=1
srun uv run python -u src/classification/hybrid_hmm/hybrid_hmm_trainer.py

# TO SYNC ADROIT WITH LOCAL - RUN FROM LOCCAL

# rsync -avz --delete \
#   --exclude '.venv' \
#   --exclude '__pycache__' \
#   --exclude '.DS_Store' \
#   --exclude '.git' \
#   ~/Documents/Github/embryo-image-processing/ \
#   jg4187@adroit.princeton.edu:/scratch/network/jg4187/embryo-image-processing/

# TO SUBMIT BATCH - RUN FROM ADROIT
# sbatch train_hmm.sh

# TO CHECK WHAT BATCHES ARE RUNNING
# squeue -u jg4187

# TO TAIL A RUNNING JOB
# cd /scratch/network/jg4187/embryo-image-processing
# tail -f logs/hmm_train_<jobid>.out
