#!/bin/bash
#SBATCH --job-name=gold_detector_aggressive
#SBATCH --output=logs/gold_detector_agg_%j.out
#SBATCH --error=logs/gold_detector_agg_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

# ============================================================================
# AGGRESSIVE CONFIGURATION (V4)
# High Recall Focus: Find almost every particle, accept more false positives
# ============================================================================

# Architecture & Model
MODEL_TYPE="${MODEL_TYPE:-unet_deep}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"

# Data Configuration (SLIDING WINDOW)
PATCH_SIZE="${PATCH_SIZE:-256}"
PATCH_STRIDE="${PATCH_STRIDE:-128}"
USE_SLIDING_WINDOW="${USE_SLIDING_WINDOW:-true}"

# Training Configuration
EPOCHS="${EPOCHS:-200}"                       # ↑ Doubled from 100 (more training time)
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-2048}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-256}"

# Learning Rate & Schedule (REDUCED for stability with high pos_weight)
LR="${LR:-1e-4}"                              # ↓ Reduced from 5e-4 (5× lower)
SCHED="${SCHED:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Loss & Target Configuration (AGGRESSIVE POSITIVE WEIGHTING)
SIGMA="${SIGMA:-1.0}"
TARGET_TYPE="${TARGET_TYPE:-gaussian}"
TARGET_RADIUS="${TARGET_RADIUS:-3}"
LOSS_TYPE="${LOSS_TYPE:-focal_bce}"
LOSS_POS_WEIGHT="${LOSS_POS_WEIGHT:-200.0}"  # ↑ Increased from 30.0 (6.7× higher)
LOSS_NEG_WEIGHT="${LOSS_NEG_WEIGHT:-1.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"

# Augmentation Configuration
USE_CLAHE="${USE_CLAHE:-false}"
SIGMA_JITTER="${SIGMA_JITTER:-true}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.1}"

# Early Stopping Configuration (MORE PATIENT)
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"   # ↑ Increased from 10 (more tolerance)
EARLY_STOP_DELTA="${EARLY_STOP_DELTA:-1e-5}"

# Training Optimization
MIXED_PRECISION="${MIXED_PRECISION:-true}"

# Data Split
POS_FRACTION="${POS_FRACTION:-0.6}"
SEED="${SEED:-42}"

SAVE_DIR="${SAVE_DIR:-checkpoints/aggressive_${SLURM_JOB_ID}}"
RESUME_PATH="${RESUME_PATH:-}"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     AGGRESSIVE GOLD PARTICLE DETECTION PIPELINE (V4)              ║"
echo "║     High Recall Focus: Find Almost Every Particle                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration Changes vs Baseline:"
echo "  Positive Weight:      $LOSS_POS_WEIGHT (was 30.0) ← 6.7× increase"
echo "  Learning Rate:        $LR (was 5e-4) ← 5× decrease"
echo "  Training Epochs:      $EPOCHS (was 100) ← 2× increase"
echo "  Early Stop Patience:  $EARLY_STOP_PATIENCE (was 10) ← More patient"
echo "  Detection Threshold:  0.08-0.10 (was 0.20) ← 50-60% lower"
echo ""
echo "Expected Performance:"
echo "  Recall:               ~96-98% (find almost all particles)"
echo "  Precision:            ~70-80% (more false positives OK)"
echo "  F1 Score:             ~0.82-0.88 (depends on threshold)"
echo "  Training Time:        ~16-18 hours (early stop around epoch 120-150)"
echo "  GPU Memory:           ~25GB"
echo ""
echo "Strategy:"
echo "  ✓ High pos_weight forces model to find particles at any cost"
echo "  ✓ Low learning rate prevents divergence with large gradients"
echo "  ✓ More epochs allow slower convergence"
echo "  ✓ Higher patience allows training to continue past plateaus"
echo "  ✓ Lower threshold captures subtle particle signals"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_detector_agg_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ============================================================================
# BUILD TRAINING COMMAND
# ============================================================================

CMD=(
  python -u train_detector.py
  --data_root "$DATA_ROOT"
  --model_type "$MODEL_TYPE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --patch_h "$PATCH_SIZE"
  --patch_w "$PATCH_SIZE"
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH"
  --sigma "$SIGMA"
  --target_type "$TARGET_TYPE"
  --target_radius "$TARGET_RADIUS"
  --base_channels "$BASE_CHANNELS"
  --loss_type "$LOSS_TYPE"
  --loss_pos_weight "$LOSS_POS_WEIGHT"
  --loss_neg_weight "$LOSS_NEG_WEIGHT"
  --focal_gamma "$FOCAL_GAMMA"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --consistency_weight "$CONSISTENCY_WEIGHT"
  --sched "$SCHED"
  --warmup_epochs "$WARMUP_EPOCHS"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --early_stop_delta "$EARLY_STOP_DELTA"
  --pos_fraction "$POS_FRACTION"
  --seed "$SEED"
  --save_dir "$SAVE_DIR"
)

# Conditional flags
if [ "$USE_CLAHE" = "true" ]; then
  CMD+=(--use_clahe)
fi

if [ "$SIGMA_JITTER" = "true" ]; then
  CMD+=(--sigma_jitter)
fi

if [ "$MIXED_PRECISION" = "true" ]; then
  CMD+=(--mixed_precision)
fi

if [ -n "$RESUME_PATH" ]; then
  CMD+=(--resume "$RESUME_PATH")
fi

# Use sliding window dataset
if [ "$USE_SLIDING_WINDOW" = "true" ]; then
  CMD+=(--use_sliding_window --patch_stride "$PATCH_STRIDE")
fi

echo "Starting aggressive training..."
echo "Command:"
echo "${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "✓ Training complete"
echo "✓ Best model saved to: $SAVE_DIR/detector_best.pt"
echo ""
echo "Next steps:"
echo "  1. Wait for training to complete (monitor with tail -f logs/gold_detector_agg_*.out)"
echo "  2. Run heatmap analysis: python analyze_heatmap_aggressive.py"
echo "  3. Evaluate with threshold sweep: python evaluate_aggressive.py"
echo "  4. Compare outputs with baseline version"
