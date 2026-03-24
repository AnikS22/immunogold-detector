#!/bin/bash

# Live monitoring script for CenterNet jobs
# Run this to watch job progress in real-time

HPC_HOST="asahai2024@athene-login.hpc.fau.edu"
JOB1=4596535  # centernet_fast (10-14h, F1 0.85-0.92)
JOB2=4596536  # centernet_nature (20-24h, F1 0.85-0.92)

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           CENTERNET LIVE JOB MONITORING                        ║"
echo "║  Job #$JOB1 (Fast) - Job #$JOB2 (Standard)                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Update interval: 10 seconds"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Monitor loop
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))

    # Clear screen
    clear

    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         CENTERNET LIVE MONITORING - Iteration $ITERATION"
    echo "║         Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Job status
    echo "📊 JOB STATUS:"
    echo "─────────────────────────────────────────────────────────────────"
    ssh $HPC_HOST "squeue -u asahai2024 -j $JOB1,$JOB2 --format='%.18i %.9P %.20j %.8u %.2t %.10M %.6D %N'" 2>/dev/null

    echo ""
    echo "📈 RECENT LOG OUTPUT:"
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    echo "Job #$JOB1 (centernet_fast) - Last 3 lines:"
    ssh $HPC_HOST "tail -3 /mnt/beegfs/home/asahai2024/max-planck-project/project/logs/centernet_*$JOB1*.out 2>/dev/null || echo 'Logs not ready yet...'" 2>/dev/null

    echo ""
    echo "Job #$JOB2 (centernet_nature) - Last 3 lines:"
    ssh $HPC_HOST "tail -3 /mnt/beegfs/home/asahai2024/max-planck-project/project/logs/centernet_*$JOB2*.out 2>/dev/null || echo 'Logs not ready yet...'" 2>/dev/null

    echo ""
    echo "═════════════════════════════════════════════════════════════════"
    echo "Next update in 10 seconds... (Ctrl+C to stop)"

    sleep 10
done
