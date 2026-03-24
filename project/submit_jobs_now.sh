#!/bin/bash

################################################################################
# CENTERNET JOB SUBMISSION SCRIPT
# Run this from your local machine to submit both jobs
################################################################################

set -e  # Exit on error

PROJECT_DIR="/Users/aniksahai/Desktop/Max Planck Project/project"
HPC_USER="asahai2024"
HPC_HOST="volta.mpibc.mpg.de"
HPC_PROJECT_DIR="/mnt/beegfs/home/asahai2024/max-planck-project/project"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        CENTERNET JOB SUBMISSION - LOCAL MACHINE                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Date: $(date)"
echo "User: $HPC_USER"
echo "Host: $HPC_HOST"
echo ""

# Verify SSH connectivity
echo "[1/4] Checking HPC connectivity..."
if ssh -q $HPC_USER@$HPC_HOST "echo 'Connected'" > /dev/null 2>&1; then
    echo "✓ HPC connection OK"
else
    echo "✗ Cannot reach HPC - check SSH access"
    echo "  Try: ssh $HPC_USER@$HPC_HOST"
    exit 1
fi

echo ""
echo "[2/4] Submitting CEM500K Baseline Job..."
JOB1=$(ssh $HPC_USER@$HPC_HOST \
    "cd $HPC_PROJECT_DIR && \
     sbatch hpc/train_centernet_cem500k.slurm" | tee /tmp/job1_output.txt)

JOB1_ID=$(echo "$JOB1" | grep -oE '[0-9]+$' | tail -1)

if [ ! -z "$JOB1_ID" ]; then
    echo "✓ Submitted: Job #$JOB1_ID"
else
    echo "✗ Failed to submit Job 1"
    cat /tmp/job1_output.txt
    exit 1
fi

echo ""
echo "[3/4] Submitting Nature-Level Accelerated Job..."
JOB2=$(ssh $HPC_USER@$HPC_HOST \
    "cd $HPC_PROJECT_DIR && \
     sbatch hpc/train_centernet_nature_fast.slurm" | tee /tmp/job2_output.txt)

JOB2_ID=$(echo "$JOB2" | grep -oE '[0-9]+$' | tail -1)

if [ ! -z "$JOB2_ID" ]; then
    echo "✓ Submitted: Job #$JOB2_ID"
else
    echo "✗ Failed to submit Job 2"
    cat /tmp/job2_output.txt
    exit 1
fi

echo ""
echo "[4/4] Checking Initial Status..."
sleep 5

ssh $HPC_USER@$HPC_HOST \
    "squeue -u asahai2024 -j $JOB1_ID,$JOB2_ID --format='%.18i %.9P %.25j %.8u %.2t %.10M %.6D %N'"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ JOBS SUBMITTED SUCCESSFULLY              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "JOB IDs:"
echo "  CEM500K Baseline:      #$JOB1_ID (18-20 hours, F1 ≈ 0.65-0.80)"
echo "  Nature-Level Accel:    #$JOB2_ID (10-14 hours, F1 ≈ 0.85-0.92)"
echo ""
echo "Expected completion: ~20 hours from now"
echo ""
echo "MONITORING COMMANDS:"
echo "  Check status:   ssh $HPC_USER@$HPC_HOST 'squeue -u asahai2024 | grep centernet'"
echo "  View logs:      ssh $HPC_USER@$HPC_HOST 'tail -50 /mnt/beegfs/.../logs/centernet_*.out'"
echo "  Check locally:  python3 $PROJECT_DIR/check_jobs.py"
echo ""
echo "═══════════════════════════════════════════════════════════════════"

# Save job info
cat > "$PROJECT_DIR/submitted_jobs_info.txt" << JOB_INFO
CENTERNET JOBS SUBMITTED
========================
Submission Time: $(date)

Job 1 - CEM500K Baseline
  ID: $JOB1_ID
  Script: hpc/train_centernet_cem500k.slurm
  Duration: 18-20 hours
  Expected F1: 0.65-0.80
  Resources: 8 CPU, 48GB, 1 GPU

Job 2 - Nature-Level Accelerated
  ID: $JOB2_ID
  Script: hpc/train_centernet_nature_fast.slurm
  Duration: 10-14 hours
  Expected F1: 0.85-0.92
  Resources: 16 CPU, 96GB, 1 GPU

Status Check:
  ssh $HPC_USER@$HPC_HOST 'squeue -u asahai2024 -j $JOB1_ID,$JOB2_ID'

Expected Completion: ~20 hours
JOB_INFO

echo "✓ Job info saved to: submitted_jobs_info.txt"
