#!/bin/bash
#SBATCH --job-name=eval_detector_opt
#SBATCH --output=logs/eval_detector_opt_%j.out
#SBATCH --error=logs/eval_detector_opt_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
cd "$PROJECT_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

# Create venv and install dependencies (reuse from training)
BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_eval_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24" -q
python -m pip install imagecodecs numpy tifffile matplotlib scipy torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q

echo "Evaluating detector from job 4594820..."
python eval_model.py
