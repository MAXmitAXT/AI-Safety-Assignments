#!/bin/bash
#SBATCH --job-name=qwen3-lora
#SBATCH --output=logs/qwen3_lora_%j.out
#SBATCH --error=logs/qwen3_lora_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Usage:
# sbatch train_lora.slurm /path/to/venv

VENV_PATH=$1

if [ -z "$VENV_PATH" ]; then
    echo "Usage: sbatch train_lora.slurm /path/to/venv"
    exit 1
fi

mkdir -p logs

source "$VENV_PATH/bin/activate"

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python train_lora_qwen.py