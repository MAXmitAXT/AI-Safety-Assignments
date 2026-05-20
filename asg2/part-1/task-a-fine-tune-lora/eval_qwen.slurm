#!/bin/bash
#SBATCH --job-name=qwen3-eval
#SBATCH --output=logs/qwen3_eval_%j.out
#SBATCH --error=logs/qwen3_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Usage:
# sbatch eval_qwen.slurm /path/to/venv [--finetuned]

VENV_PATH=$1
MODEL_FLAG=$2

if [ -z "$VENV_PATH" ]; then
    echo "Usage: sbatch eval_qwen.slurm /path/to/venv [--finetuned]"
    exit 1
fi

mkdir -p logs

module load gcc
module load cuda

source "$VENV_PATH/bin/activate"

echo "========== GPU CHECK =========="
hostname
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"
echo "================================"

python -u evaluate_qwen.py $MODEL_FLAG