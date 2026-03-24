#!/bin/bash
#SBATCH --job-name=centernet_cem500k
#SBATCH --output=logs/centernet_cem500k_%j.out
#SBATCH --error=logs/centernet_cem500k_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     CENTERNET CEM500K TRANSFER LEARNING BASELINE                  ║"
echo "║     Particle Detection with Transfer Learning from 500K EM Images ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Architecture & Training Strategy:"
echo "  Model:              CenterNet (CEM500K ResNet50 + FPN)"
echo "  Encoder:            CEM500K pre-trained (500K EM images)"
echo "  Training Strategy:  Phase 1 (frozen): 1-15 epochs"
echo "                      Phase 2 (unfrozen): 16-100 epochs"
echo ""
echo "Expected Performance:"
echo "  Baseline CenterNet:             F1 ≈ 0.65"
echo "  + Class imbalance + Boundary:   F1 ≈ 0.80"
echo "  + Contrastive pre-training:     F1 ≈ 0.85-0.88"
echo "  + Full improvements ensemble:   F1 ≈ 0.95+ (NATURE-LEVEL)"
echo ""
echo "Dataset:"
echo "  • 10 EM images (2048×2115 pixels)"
echo "  • 344 particles total (316×6nm, 28×12nm)"
echo "  • Train: 7 images, Val: 2 images, Test: 1 image"
echo "  • Sliding window: 256×256 patches, stride=128"
echo "  • Training samples/epoch: 2048"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs checkpoints

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.4 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_centernet_cem500k_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24" -q || true

echo "Installing dependencies..."
python -m pip install -q imagecodecs numpy tifffile scipy matplotlib scikit-image pandas 2>/dev/null || true
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
python -m pip install -q timm 2>/dev/null || true

echo ""
echo "Starting CenterNet CEM500K baseline training..."
echo ""

python -u train_centernet.py \
  --data_root "$DATA_ROOT" \
  --patch_h 256 \
  --patch_w 256 \
  --patch_stride 128 \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-4 \
  --lr_encoder 1e-5 \
  --weight_decay 1e-4 \
  --freeze_encoder_epochs 15 \
  --unfreeze_blocks 2 \
  --sigma 1.0 \
  --warmup_epochs 5 \
  --early_stop_patience 15 \
  --early_stop_delta 1e-5 \
  --seed 42 \
  --save_dir "checkpoints/centernet_cem500k_${SLURM_JOB_ID}" \
  --num_workers 7 \
  --mixed_precision \
  --gradient_centralization

# Note: --mixed_precision keeps forward in FP16; loss is computed in FP32 (see train_centernet.py).

echo ""
echo "✓ Training complete"
echo "✓ Best model saved to: checkpoints/centernet_cem500k_${SLURM_JOB_ID}/detector_best.pt"
