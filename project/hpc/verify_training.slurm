#!/bin/bash
#SBATCH --job-name=verify_training
#SBATCH --output=logs/verify_training_%j.out
#SBATCH --error=logs/verify_training_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

python3 eval_simple.py
