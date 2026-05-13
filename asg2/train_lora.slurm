#!/bin/bash
#SBATCH --job-name=qwen3-lora
#SBATCH --output=logs/qwen3_lora_%j.out
#SBATCH --error=logs/qwen3_lora_%j.err
#SBATCH --time=12:00:00

#SBATCH -p grete:shared
#SBATCH -G A100:1

# Optional: request 80GB A100 if needed
##SBATCH --constraint=80gb

VENV_PATH=$1

if [ -z "$VENV_PATH" ]; then
    echo "Usage: sbatch train_lora.slurm /path/to/venv"
    exit 1
fi

mkdir -p logs

module load gcc
module load cuda

source "$VENV_PATH/bin/activate"

export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Running on node: $SLURM_NODELIST"
python --version
python -u train_lora_qwen.py