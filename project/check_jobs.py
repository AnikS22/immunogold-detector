#!/usr/bin/env python3
"""
CenterNet Job Status Checker
Monitors training jobs and retrieves status/logs
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Command timeout"
    except Exception as e:
        return f"Error: {e}"

def check_job_status(job_ids=None):
    """Check status of CenterNet jobs"""
    print("="*70)
    print("CENTERNET JOB STATUS CHECK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Get all centernet jobs
    cmd = "squeue -u asahai2024 -h | grep centernet"
    output = run_command(cmd)

    if not output.strip():
        print("❌ No CenterNet jobs found in queue")
        print("   Jobs may have completed or failed")
        print("\nChecking completed jobs:")
        cmd = "sacct -u asahai2024 -S now-1day | grep centernet"
        completed = run_command(cmd)
        if completed.strip():
            print(completed)
        return False
    else:
        print("\n✓ CenterNet jobs running:")
        print(output)
        return True

def check_logs():
    """Check training logs"""
    print("\n" + "="*70)
    print("TRAINING LOGS")
    print("="*70)

    log_dir = Path("/Users/aniksahai/Desktop/Max Planck Project/project/logs")
    if not log_dir.exists():
        print("⚠ Log directory not found locally (on HPC)")
        print("Access logs via: ssh asahai2024@volta.mpibc.mpg.de")
        return

    centernet_logs = list(log_dir.glob("centernet_*.out"))
    if not centernet_logs:
        print("No logs found yet")
        return

    for log_file in sorted(centernet_logs, reverse=True)[:2]:
        print(f"\n📄 {log_file.name} (Last 10 lines):")
        print("-" * 70)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())

def check_memory_gpu():
    """Check GPU memory usage"""
    print("\n" + "="*70)
    print("GPU & MEMORY STATUS")
    print("="*70)

    cmd = "srun -u asahai2024 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null"
    output = run_command(cmd)

    if output.strip() and "error" not in output.lower():
        print("GPU Status:")
        print(output)
    else:
        print("⚠ Unable to retrieve GPU status")

def check_checkpoints():
    """Check if checkpoints exist"""
    print("\n" + "="*70)
    print("CHECKPOINT STATUS")
    print("="*70)

    checkpoint_dir = Path("/Users/aniksahai/Desktop/Max Planck Project/project/checkpoints")
    if not checkpoint_dir.exists():
        print("Checkpoint directory not created yet")
        return

    for subdir in sorted(checkpoint_dir.glob("centernet_*/"))[:3]:
        checkpoint_file = subdir / "detector_best.pt"
        if checkpoint_file.exists():
            size_mb = checkpoint_file.stat().st_size / (1024*1024)
            mtime = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
            print(f"✓ {subdir.name}/detector_best.pt ({size_mb:.1f} MB) - Updated: {mtime}")
        else:
            print(f"⏳ {subdir.name}/ - Training in progress...")

def main():
    """Main monitoring function"""
    print("\n")

    # Check job status
    running = check_job_status()

    if running:
        # If jobs running, check additional info
        check_logs()
        check_memory_gpu()
        check_checkpoints()

        print("\n" + "="*70)
        print("✓ Monitoring continuing...")
        print("  Run this script again to check updated status")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠ No active jobs")
        print("  Check logs for completion status:")
        print("  - tail -50 logs/centernet_*.out")
        print("="*70)

if __name__ == "__main__":
    main()
