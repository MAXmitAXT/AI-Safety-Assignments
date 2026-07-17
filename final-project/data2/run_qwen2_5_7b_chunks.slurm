#!/bin/bash
#SBATCH --job-name=qwen2-5-7b
#SBATCH --account=kisski-asc2026
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/qwen2_5_7b_chunk_%A_%a.out
#SBATCH --error=logs/qwen2_5_7b_chunk_%A_%a.err

source /mnt/vast-kisski/projects/kisski-asc2026/users/u29088/venv/bin/activate
cd /mnt/vast-kisski/projects/kisski-asc2026/tue-ai-safety-course/cluster-gwdg

export HF_HOME=/mnt/vast-kisski/projects/kisski-asc2026/users/u29088/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

MODEL_DIR=$(find $HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
IDX=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python run_qwen_jsonl.py \
  --model "$MODEL_DIR" \
  --input "chunks/qwen2_5_7b_chunk_${IDX}.jsonl" \
  --output "outputs/qwen2_5_7b_chunk_${IDX}.jsonl" \
  --num-samples 10

