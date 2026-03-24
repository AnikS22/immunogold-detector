#!/bin/bash

# CenterNet Job Monitoring Script
# Monitors training progress on HPC

PROJECT_DIR="/Users/aniksahai/Desktop/Max Planck Project/project"
LOG_FILE="$PROJECT_DIR/job_monitoring.log"

echo "CenterNet Job Monitor Started at $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Function to check job status
check_jobs() {
    echo ""
    echo "=== Status Check: $(date) ===" >> "$LOG_FILE"

    # Check both jobs
    squeue -u asahai2024 | grep centernet >> "$LOG_FILE" 2>&1

    # Get job details if running
    RUNNING=$(squeue -u asahai2024 -h | grep centernet | wc -l)

    if [ $RUNNING -gt 0 ]; then
        echo "✓ Jobs still running ($RUNNING active)" >> "$LOG_FILE"

        # Try to get loss values from logs
        echo "" >> "$LOG_FILE"
        echo "Recent training progress:" >> "$LOG_FILE"

        # Check for log files and tail them
        for log in $(find $PROJECT_DIR/logs -name "centernet_*.out" -newer /tmp/centernet_last_check 2>/dev/null); do
            echo "From $log:" >> "$LOG_FILE"
            tail -5 "$log" >> "$LOG_FILE" 2>/dev/null
        done
    else
        echo "⚠ No running CenterNet jobs found" >> "$LOG_FILE"
    fi

    touch /tmp/centernet_last_check
}

# Monitor loop
echo "Starting monitoring..."
echo "Logs written to: $LOG_FILE"
echo ""

# Check every 5 minutes for first hour, then every 15 minutes
COUNT=0
while true; do
    check_jobs

    # Display current status
    echo "Status update: $(date '+%Y-%m-%d %H:%M:%S')"
    squeue -u asahai2024 | grep centernet | head -5

    # Sleep duration based on elapsed time
    if [ $COUNT -lt 12 ]; then
        # First hour: check every 5 minutes
        sleep 300
    else
        # After first hour: check every 15 minutes
        sleep 900
    fi

    COUNT=$((COUNT + 1))

    # Exit after 24 hours (in case we run forever)
    if [ $COUNT -gt 96 ]; then
        echo "Monitoring ended after 24 hours" >> "$LOG_FILE"
        break
    fi
done
