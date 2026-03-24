#!/bin/bash
#SBATCH --job-name=eval_centernet_pair
#SBATCH --output=logs/eval_centernet_pair_fixed_%j.out
#SBATCH --error=logs/eval_centernet_pair_fixed_%j.err
#SBATCH --partition=longq7-eng
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/beegfs/home/asahai2024/max-planck-project/project}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses}"
OUT_DIR="${OUT_DIR:-hpc_eval_centernet}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR" logs

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.12.4 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_eval_centernet_pair_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade "pip<24" || true
python -m pip install -q imagecodecs numpy tifffile scipy scikit-image pandas 2>/dev/null || true
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
python -m pip install -q timm 2>/dev/null || true

python eval_centernet_infer.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$PROJECT_DIR/checkpoints/centernet_cem500k_4597168/detector_best.pt" \
  --out_csv "$OUT_DIR/centernet_cem500k_4597168.csv" \
  --tile 512 --stride 256 --pre_threshold 0.02 --min_distance 4 --max_det_per_class 500

python eval_centernet_infer.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$PROJECT_DIR/checkpoints/centernet_nature_fast_4597169/detector_best.pt" \
  --out_csv "$OUT_DIR/centernet_nature_fast_4597169.csv" \
  --tile 512 --stride 256 --pre_threshold 0.02 --min_distance 4 --max_det_per_class 500

python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/centernet_cem500k_4597168.csv" \
  --match_dist 15.0 \
  --threshold_sweep "0.0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60" \
  > "$OUT_DIR/centernet_cem500k_4597168_eval.txt"

python evaluate_detector.py \
  --data_root "$DATA_ROOT" \
  --pred_csv "$OUT_DIR/centernet_nature_fast_4597169.csv" \
  --match_dist 15.0 \
  --threshold_sweep "0.0,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60" \
  > "$OUT_DIR/centernet_nature_fast_4597169_eval.txt"

echo "===== centernet_cem500k_4597168 ====="
rg "^(all\\.f1=|class_6nm\\.f1=|class_12nm\\.f1=|macro\\.f1=|best_threshold=)" "$OUT_DIR/centernet_cem500k_4597168_eval.txt" | tail -20
echo "===== centernet_nature_fast_4597169 ====="
rg "^(all\\.f1=|class_6nm\\.f1=|class_12nm\\.f1=|macro\\.f1=|best_threshold=)" "$OUT_DIR/centernet_nature_fast_4597169_eval.txt" | tail -20
