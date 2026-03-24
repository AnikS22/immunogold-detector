#!/usr/bin/env bash
# Sync immunogold detector code to FAU Athene and submit CenterNet training jobs.
#
# Usage (from your Mac, in a normal terminal):
#   chmod +x hpc/submit_centernet_training_athene.sh
#   ./hpc/submit_centernet_training_athene.sh
#
# Requires: SSH key to Athene, data already under REMOTE .../project/data/...
#   (If not, rsync your "analyzed synapses" folder once, or set SYNC_DATA=1 — slow.)
#
set -euo pipefail

# Optional: run local preflight before rsync (requires: pip install timm; use same --data-root as on cluster)
#   VALIDATE=1 ./hpc/submit_centernet_training_athene.sh
#   VALIDATE=0 ./hpc/submit_centernet_training_athene.sh   # skip checks
VALIDATE="${VALIDATE:-0}"

REMOTE_HOST="${REMOTE_HOST:-asahai2024@athene-login.hpc.fau.edu}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/beegfs/home/asahai2024/max-planck-project}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REMOTE_PROJECT_DIR="$REMOTE_BASE/project"
SSH_OPTS="${SSH_OPTS:--o ServerAliveInterval=30 -o ServerAliveCountMax=10}"

# Set SYNC_DATA=1 to also push local data/ (large). Default: only code + configs.
SYNC_DATA="${SYNC_DATA:-0}"

if [[ "$VALIDATE" == "1" ]]; then
  echo "Running validate_before_slurm.py (set VALIDATE=0 to skip)..."
  if [[ -n "${VALIDATE_DATA_ROOT:-}" ]]; then
    (cd "$LOCAL_PROJECT_DIR" && python3 validate_before_slurm.py --data-root "$VALIDATE_DATA_ROOT") || {
      echo "Preflight failed. Fix errors, install deps (pip install timm), or run with VALIDATE=0"
      exit 1
    }
  else
    (cd "$LOCAL_PROJECT_DIR" && python3 validate_before_slurm.py) || {
      echo "Preflight failed. Fix errors, install deps (pip install timm), or run with VALIDATE=0"
      exit 1
    }
  fi
fi

echo "=============================================="
echo "  Athene: CenterNet training submission"
echo "  Local:  $LOCAL_PROJECT_DIR"
echo "  Remote: $REMOTE_HOST:$REMOTE_PROJECT_DIR"
echo "=============================================="

ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p \"$REMOTE_PROJECT_DIR/logs\" \"$REMOTE_PROJECT_DIR/checkpoints\""

echo ""
echo "[1/3] Rsync project (excluding venv, checkpoints, caches)..."
rsync -avz --progress \
  -e "ssh $SSH_OPTS" \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "venv/" \
  --exclude ".venv/" \
  --exclude "checkpoints/" \
  --exclude "predictions/" \
  --exclude "*.pt" \
  --exclude "data/" \
  "$LOCAL_PROJECT_DIR/" \
  "$REMOTE_HOST:$REMOTE_PROJECT_DIR/"

if [[ "$SYNC_DATA" == "1" ]]; then
  echo ""
  echo "[optional] SYNC_DATA=1 — syncing data/ (this can take a long time)..."
  rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    "$LOCAL_PROJECT_DIR/data/" \
    "$REMOTE_HOST:$REMOTE_PROJECT_DIR/data/"
fi

DATA_ROOT="$REMOTE_PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses"

echo ""
echo "[2/3] Submitting CenterNet CEM500k baseline (train_centernet.py)..."
OUT1=$(ssh $SSH_OPTS "$REMOTE_HOST" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT_DIR"
export DATA_ROOT="$DATA_ROOT"
export PROJECT_DIR="$REMOTE_PROJECT_DIR"
sbatch hpc/train_centernet_cem500k.slurm
EOF
)
echo "$OUT1"
JOB1=$(echo "$OUT1" | awk '/Submitted batch job/ {print $4}')

echo ""
echo "[3/3] Submitting CenterNet nature-fast (train_centernet_enhanced.py)..."
OUT2=$(ssh $SSH_OPTS "$REMOTE_HOST" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT_DIR"
export DATA_ROOT="$DATA_ROOT"
export PROJECT_DIR="$REMOTE_PROJECT_DIR"
sbatch hpc/train_centernet_nature_fast.slurm
EOF
)
echo "$OUT2"
JOB2=$(echo "$OUT2" | awk '/Submitted batch job/ {print $4}')

echo ""
echo "Submitted (save these IDs):"
echo "  CEM500k baseline:  ${JOB1:-?}"
echo "  Nature fast:       ${JOB2:-?}"
echo ""
echo "Monitor:"
echo "  ssh $REMOTE_HOST 'squeue -u \\\$USER | grep centernet'"
echo ""
echo "Logs (after a minute):"
echo "  ssh $REMOTE_HOST 'tail -f $REMOTE_PROJECT_DIR/logs/centernet_cem500k_${JOB1}.out'"
echo "  ssh $REMOTE_HOST 'tail -f $REMOTE_PROJECT_DIR/logs/centernet_nature_fast_${JOB2}.out'"
echo ""
echo "If jobs exit immediately, verify data exists:"
echo "  ssh $REMOTE_HOST 'ls \"$DATA_ROOT\" | head'"
