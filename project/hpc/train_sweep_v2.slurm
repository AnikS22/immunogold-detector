#!/bin/bash
#SBATCH --job-name=gold_v2
#SBATCH --output=logs/gold_v2_%A_%a.out
#SBATCH --error=logs/gold_v2_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19

# ============================================================================
# V2 SWEEP — 20 jobs with fixed coordinates, S1 palette, S4 transpose
# Submit with:  sbatch hpc/train_sweep_v2.slurm
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"

# Format: MODEL|LOSS|SIGMA|POS_WEIGHT|LR|EPOCHS|PATCH_H|PATCH_STRIDE|BASE_CH|EXTRA_FLAGS|LABEL
#
# 20 jobs organized by what question each answers:
#
# Jobs 0-4:   ARCHITECTURE — deep vs shallow, channel width
# Jobs 5-8:   SIGMA — how big should heatmap targets be?
# Jobs 9-11:  LOSS — focal BCE vs fixed weighted MSE, pos_weight
# Jobs 12-14: PREPROCESSING — mantis contrast, sigma jitter, CLAHE
# Jobs 15-16: PATCH SIZE — 256 vs 512
# Jobs 17-18: TARGET TYPE — gaussian vs disk
# Job 19:     KITCHEN SINK — best guesses combined
#
CONFIGS=(
  # --- ARCHITECTURE (which model learns best from 453 particles?) ---
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|32||deep_baseline"
  "unet|focal_bce|2.0|300|3e-4|200|256|128|32||shallow_baseline"
  "unet|focal_bce|2.0|300|3e-4|200|256|128|64||shallow_ch64"
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|64||deep_ch64"
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|16||deep_ch16"

  # --- SIGMA (how wide should target Gaussians be?) ---
  "unet_deep|focal_bce|1.0|300|3e-4|200|256|128|32||sigma1.0"
  "unet_deep|focal_bce|1.5|300|3e-4|200|256|128|32||sigma1.5"
  "unet_deep|focal_bce|3.0|300|3e-4|200|256|128|32||sigma3.0"
  "unet_deep|focal_bce|4.0|300|3e-4|200|256|128|32||sigma4.0"

  # --- LOSS FUNCTION (focal BCE vs weighted MSE, pos_weight) ---
  "unet_deep|weighted_mse|2.0|300|3e-4|200|256|128|32||wmse_pw300"
  "unet_deep|focal_bce|2.0|500|3e-4|200|256|128|32||fbce_pw500"
  "unet_deep|focal_bce|2.0|100|3e-4|200|256|128|32||fbce_pw100"

  # --- PREPROCESSING (mantis, sigma jitter, both) ---
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|32|--use_mantis|mantis"
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|32|--sigma_jitter|sigma_jitter"
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|32|--use_mantis --sigma_jitter|mantis_jitter"

  # --- PATCH SIZE (256 vs 512 — more context per patch) ---
  "unet_deep|focal_bce|2.0|300|3e-4|200|512|256|32||patch512"
  "unet_deep|focal_bce|2.0|300|3e-4|200|512|256|32|--use_mantis|patch512_mantis"

  # --- TARGET TYPE (gaussian blobs vs binary disks) ---
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|32|--target_type disk --target_radius 3|disk_r3"
  "unet_deep|focal_bce|2.0|300|3e-4|200|256|128|32|--target_type disk --target_radius 5|disk_r5"

  # --- KITCHEN SINK (combine what we think is best) ---
  "unet_deep|focal_bce|2.5|500|3e-4|300|256|128|32|--use_mantis --sigma_jitter|best_guess"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r MODEL_TYPE LOSS_TYPE SIGMA POS_WEIGHT LR EPOCHS PATCH_H PATCH_STRIDE BASE_CH EXTRA_FLAGS LABEL <<< "$CONFIG"

SAVE_DIR="checkpoints/v2_${SLURM_ARRAY_JOB_ID}_${LABEL}"

echo "================================================================"
echo "  V2 SWEEP JOB $SLURM_ARRAY_TASK_ID / 19: $LABEL"
echo "  model=$MODEL_TYPE  loss=$LOSS_TYPE"
echo "  sigma=$SIGMA  pos_weight=$POS_WEIGHT  lr=$LR  epochs=$EPOCHS"
echo "  patch=${PATCH_H}  stride=$PATCH_STRIDE  channels=$BASE_CH"
echo "  extra: $EXTRA_FLAGS"
echo "================================================================"

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR" "$SAVE_DIR/vis"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_v2_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy Pillow
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ============================================================
# PHASE 1: TRAIN
# ============================================================
echo ""
echo "=== PHASE 1: TRAINING ==="
python -u train_detector.py \
  --data_root "$DATA_ROOT" \
  --model_type "$MODEL_TYPE" \
  --base_channels "$BASE_CH" \
  --epochs "$EPOCHS" \
  --batch_size 8 \
  --lr "$LR" \
  --patch_h "$PATCH_H" \
  --patch_w "$PATCH_H" \
  --train_samples_per_epoch 2048 \
  --val_samples_per_epoch 256 \
  --sigma "$SIGMA" \
  --loss_type "$LOSS_TYPE" \
  --loss_pos_weight "$POS_WEIGHT" \
  --focal_gamma 2.0 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --sched cosine \
  --warmup_epochs 10 \
  --early_stop_patience 25 \
  --early_stop_delta 1e-6 \
  --seed 42 \
  --save_dir "$SAVE_DIR" \
  --use_sliding_window \
  --patch_stride "$PATCH_STRIDE" \
  $EXTRA_FLAGS

echo ""
echo "Training complete."

# ============================================================
# PHASE 2: INFERENCE (low threshold to capture all detections)
# ============================================================
echo ""
echo "=== PHASE 2: INFERENCE ==="
python -u infer_detector.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$SAVE_DIR/detector_best.pt" \
  --model_type "$MODEL_TYPE" \
  --base_channels "$BASE_CH" \
  --out_csv "$SAVE_DIR/predictions.csv" \
  --out_vis_dir "$SAVE_DIR/vis" \
  --tile_h "$PATCH_H" \
  --tile_w "$PATCH_H" \
  --stride_h "$PATCH_STRIDE" \
  --stride_w "$PATCH_STRIDE" \
  --threshold 0.01 \
  --min_distance 3 \
  --save_vis

# ============================================================
# PHASE 3: EVALUATE WITH THRESHOLD SWEEP
# ============================================================
echo ""
echo "=== PHASE 3: EVALUATION ==="

# Sweep thresholds to find optimal
python -u evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$SAVE_DIR/predictions.csv" \
  --match_dist 10 \
  --sweep_start 0.01 \
  --sweep_end 0.5 \
  --sweep_steps 50 \
  | tee "$SAVE_DIR/eval_results.txt"

# Also evaluate at match_dist=5 for stricter localization
echo ""
echo "=== STRICT EVALUATION (5px) ==="
python -u evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$SAVE_DIR/predictions.csv" \
  --match_dist 5 \
  --sweep_start 0.01 \
  --sweep_end 0.5 \
  --sweep_steps 50 \
  | tee "$SAVE_DIR/eval_strict.txt"

echo ""
echo "================================================================"
echo "  DONE — $LABEL"
echo "  Results: $SAVE_DIR/eval_results.txt"
echo "  Strict:  $SAVE_DIR/eval_strict.txt"
echo "  Visuals: $SAVE_DIR/vis/"
echo "================================================================"
