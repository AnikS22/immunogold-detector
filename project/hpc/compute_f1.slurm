#!/bin/bash
#SBATCH --job-name=compute_f1
#SBATCH --output=logs/f1_%j.out
#SBATCH --error=logs/f1_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

# Create minimal venv (reuse from training if possible)
BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_f1_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24" -q 2>/dev/null || true
python -m pip install numpy scipy torch torchvision tifffile --index-url https://download.pytorch.org/whl/cu118 -q 2>/dev/null || true

echo "Computing F1 score..."
python compute_f1.py
