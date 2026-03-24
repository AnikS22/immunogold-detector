#!/usr/bin/env bash
# Run all fast checks before rsync/sbatch. Exit non-zero if anything fails.
#
# Usage (from repo):
#   cd project
#   ./hpc/validate_before_submit.sh
#   ./hpc/validate_before_submit.sh --data-root "data/Max Planck Data/Gold Particle Labelling/analyzed synapses"
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "Running validate_before_slurm.py ..."
python3 validate_before_slurm.py "$@"
