#!/bin/bash
#SBATCH --job-name=gold_detector_cem500k
#SBATCH --output=logs/gold_detector_cem500k_%j.out
#SBATCH --error=logs/gold_detector_cem500k_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

# ============================================================================
# CEM500K TRANSFER LEARNING CONFIGURATION
# ============================================================================

# Architecture & Model (CEM500K ResNet50 encoder)
MODEL_TYPE="${MODEL_TYPE:-cem500k}"
PRETRAINED="${PRETRAINED:-true}"

# Data Configuration (same as optimized pipeline)
PATCH_SIZE="${PATCH_SIZE:-256}"
PATCH_STRIDE="${PATCH_STRIDE:-128}"

# Training Configuration
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-256}"

# Transfer Learning Configuration
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-15}"
UNFREEZE_BLOCKS="${UNFREEZE_BLOCKS:-2}"

# Learning Rates (conservative for transfer learning)
LR="${LR:-1e-4}"                          # Decoder learning rate
LR_ENCODER="${LR_ENCODER:-1e-5}"          # Encoder learning rate (5× smaller)
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Loss & Target Configuration
SIGMA="${SIGMA:-1.0}"                     # Same as optimized pipeline
TARGET_TYPE="${TARGET_TYPE:-gaussian}"
LOSS_TYPE="${LOSS_TYPE:-focal_bce}"
LOSS_POS_WEIGHT="${LOSS_POS_WEIGHT:-30.0}"
LOSS_NEG_WEIGHT="${LOSS_NEG_WEIGHT:-1.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"

# Augmentation Configuration
SIGMA_JITTER="${SIGMA_JITTER:-true}"

# Early Stopping Configuration (longer patience for transfer learning)
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-15}"
EARLY_STOP_DELTA="${EARLY_STOP_DELTA:-1e-5}"

# Training Optimization
MIXED_PRECISION="${MIXED_PRECISION:-true}"

# Data Split
SEED="${SEED:-42}"

SAVE_DIR="${SAVE_DIR:-checkpoints/${SLURM_JOB_ID}}"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     CEM500K TRANSFER LEARNING - GOLD PARTICLE DETECTION           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Model:                CEM500K ResNet50 + UNet (28M params)"
echo "  Encoder:              CEM500K pre-trained (500K EM images via MoCoV2)"
echo "  Decoder:              4-level symmetric with skip connections"
echo "  Data Strategy:        Sliding Window ($PATCH_SIZE×$PATCH_SIZE, stride=$PATCH_STRIDE)"
echo ""
echo "Transfer Learning Strategy:"
echo "  Phase 1 (epochs 1-$FREEZE_ENCODER_EPOCHS):  Frozen encoder, train decoder only"
echo "                                  LR: $LR"
echo "  Phase 2 (epochs $((FREEZE_ENCODER_EPOCHS+1))-$EPOCHS): Unfreeze last $UNFREEZE_BLOCKS blocks"
echo "                                  Decoder LR: $LR"
echo "                                  Encoder LR: $LR_ENCODER"
echo ""
echo "Expected Improvements:"
echo "  vs Optimized (V3):    5-10× F1 improvement (0.01 → 0.05-0.10)"
echo "  Rationale:            CEM500K pre-training captures EM-specific features"
echo "  Training Time:        15-20 hours (with early stopping)"
echo "  GPU Memory:           ~22-26GB for batch_size=8"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.4 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_detector_cem500k_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24" -q || true

echo "Installing packages..."
python -m pip install -q imagecodecs numpy tifffile matplotlib scipy 2>/dev/null || true
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
python -m pip install -q timm 2>/dev/null || true

# ============================================================================
# BUILD TRAINING COMMAND
# ============================================================================

CMD=(
  python -u train_detector_cem500k.py
  --data_root "$DATA_ROOT"
  --patch_h "$PATCH_SIZE"
  --patch_w "$PATCH_SIZE"
  --patch_stride "$PATCH_STRIDE"
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --lr_encoder "$LR_ENCODER"
  --weight_decay "$WEIGHT_DECAY"
  --freeze_encoder_epochs "$FREEZE_ENCODER_EPOCHS"
  --unfreeze_blocks "$UNFREEZE_BLOCKS"
  --sigma "$SIGMA"
  --loss_type "$LOSS_TYPE"
  --loss_pos_weight "$LOSS_POS_WEIGHT"
  --loss_neg_weight "$LOSS_NEG_WEIGHT"
  --focal_gamma "$FOCAL_GAMMA"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --early_stop_delta "$EARLY_STOP_DELTA"
  --seed "$SEED"
  --save_dir "$SAVE_DIR"
)

# Conditional flags
if [ "$SIGMA_JITTER" = "true" ]; then
  CMD+=(--sigma_jitter)
fi

if [ "$MIXED_PRECISION" = "true" ]; then
  CMD+=(--mixed_precision)
fi

echo "Starting CEM500K transfer learning training..."
echo "Command:"
echo "${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "✓ Training complete"
echo "✓ Best model saved to: $SAVE_DIR/detector_best.pt"
