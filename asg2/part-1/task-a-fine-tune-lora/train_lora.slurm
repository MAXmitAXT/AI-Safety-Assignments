#!/bin/bash
#SBATCH --job-name=qwen3-lora
#SBATCH --output=logs/qwen3_lora_%j.out
#SBATCH --error=logs/qwen3_lora_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=kisski
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

VENV_PATH=$1

mkdir -p logs

module load gcc
module load cuda

source "$VENV_PATH/bin/activate"

echo "========== GPU CHECK =========="
hostname
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
echo "==============================="

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

python -u train_lora_qwen.py