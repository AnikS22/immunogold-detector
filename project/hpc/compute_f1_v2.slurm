#!/bin/bash
#SBATCH --job-name=compute_f1
#SBATCH --output=logs/f1_%j.out
#SBATCH --error=logs/f1_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail

PROJECT_DIR="/mnt/beegfs/home/asahai2024/max-planck-project/project"
cd "$PROJECT_DIR"

module purge 2>/dev/null || true
module load python3/3.9.16 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

# Setup venv with error checking
VENV_DIR="/mnt/beegfs/home/asahai2024/venv_f1"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Upgrading pip..."
python -m pip install --upgrade "pip<24"
echo "Installing packages..."
python -m pip install numpy scipy torch torchvision tifffile

echo ""
echo "Computing F1 score..."
python compute_f1.py
