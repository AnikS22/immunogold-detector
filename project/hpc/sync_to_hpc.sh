#!/usr/bin/env bash
# Push local immunogold repo to HPC (rsync). Does not submit jobs.
#
# Usage (from Mac, on VPN if required):
#   ./hpc/sync_to_hpc.sh
#
# Athene (default):
#   ./hpc/sync_to_hpc.sh
#
# MPI Volta:
#   REMOTE_HOST=asahai2024@volta.mpibc.mpg.de ./hpc/sync_to_hpc.sh
#
# Include dataset (large):
#   SYNC_DATA=1 ./hpc/sync_to_hpc.sh
#
# Include checkpoints (very large):
#   SYNC_CHECKPOINTS=1 ./hpc/sync_to_hpc.sh
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-asahai2024@athene-login.hpc.fau.edu}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/beegfs/home/asahai2024/max-planck-project}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root: .../Max Planck Project (parent of project/)
LOCAL_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSH_OPTS="${SSH_OPTS:--o ServerAliveInterval=30 -o ServerAliveCountMax=120 -o ConnectTimeout=30}"

SYNC_DATA="${SYNC_DATA:-0}"
SYNC_CHECKPOINTS="${SYNC_CHECKPOINTS:-0}"

echo "============================================================"
echo "  Sync local repo → HPC"
echo "  Remote: $REMOTE_HOST:$REMOTE_BASE"
echo "  Local:  $LOCAL_REPO_ROOT"
echo "  SYNC_DATA=${SYNC_DATA}  SYNC_CHECKPOINTS=${SYNC_CHECKPOINTS}"
echo "============================================================"

ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p \"$REMOTE_BASE/project/logs\" \"$REMOTE_BASE/project/checkpoints\""

RSYNC_EXCLUDES=(
  "--exclude=.git/"
  "--exclude=**/.DS_Store"
  "--exclude=**/__pycache__/"
  "--exclude=*.pyc"
  "--exclude=**/.venv/"
  "--exclude=**/venv/"
  "--exclude=.idea/"
  "--exclude=.vscode/"
)

if [[ "$SYNC_DATA" != "1" ]]; then
  RSYNC_EXCLUDES+=("--exclude=project/data/")
fi
if [[ "$SYNC_CHECKPOINTS" != "1" ]]; then
  RSYNC_EXCLUDES+=("--exclude=project/checkpoints/")
fi

echo ""
echo "Running rsync..."
rsync -avz --progress --partial --inplace \
  -e "ssh $SSH_OPTS" \
  "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_REPO_ROOT/" \
  "$REMOTE_HOST:$REMOTE_BASE/"

echo ""
echo "Done. On the cluster:"
echo "  cd $REMOTE_BASE/project"
echo "  # optional: git status  (if you also use git on HPC)"
echo "============================================================"
