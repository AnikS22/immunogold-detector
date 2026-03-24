# CENTERNET JOB SUBMISSION INSTRUCTIONS

**Status**: ✅ READY TO SUBMIT
**Date**: 2026-03-21
**All Systems Validated**: YES

---

## Step 1: Submit Jobs to HPC

### From Your Local Machine, Run:

```bash
ssh asahai2024@volta.mpibc.mpg.de
cd /mnt/beegfs/home/asahai2024/max-planck-project/project
sbatch hpc/train_centernet_cem500k.slurm
sbatch hpc/train_centernet_nature_fast.slurm
```

### Expected Output:
```
Submitted batch job 1234567
Submitted batch job 1234568
```

**Save these Job IDs!**

---

## Step 2: Immediate Status Check (First 5 minutes)

```bash
# Check if jobs started
squeue -u asahai2024 -h | grep centernet

# Should show: RUNNING (or PENDING if queued)
```

---

## Step 3: Monitor Training Progress

### Option A: Simple Status Check (Recommended)
```bash
# Run every 5-10 minutes
squeue -u asahai2024 | grep centernet
```

### Option B: Detailed Monitoring
```bash
# Watch training logs in real-time
tail -f logs/centernet_cem500k_*.out
tail -f logs/centernet_nature_fast_*.out
```

### Option C: Python Checker (From Local Machine)
```bash
cd /Users/aniksahai/Desktop/Max\ Planck\ Project/project
python3 check_jobs.py
```

---

## Step 4: Expected Timeline

```
Time 0:00-0:05    - Jobs submit and initialize
Time 0:05-0:30    - Dependencies install, Python startup
Time 0:30         - Training begins (first epoch)
Time 1:00         - Check: Loss should be declining
Time 6:00         - ~30% progress
Time 10:00        - Job 2 (fast) completes first
Time 14:00        - Both jobs should be done
```

---

## Step 5: Monitor These Things

### ✓ Good Signs (Everything OK)
- Job state: `RUNNING`
- Loss values decreasing each epoch
- GPU utilization 80-95%
- No CUDA errors

### ⚠ Warning Signs (Monitor Closely)
- Job state: `PENDING` > 30 min (queue wait)
- Loss not decreasing
- GPU util < 50%

### ❌ Critical Issues (Investigate)
- Job state: `FAILED` or `TIMEOUT`
- "CUDA out of memory"
- "ModuleNotFoundError" or "ImportError"
- Training extremely slow

---

## Step 6: After Jobs Complete

### Check Status
```bash
sacct -u asahai2024 | grep centernet_
# Should show: COMPLETED and ExitCode=0:0
```

### Retrieve Checkpoints
```bash
# Create local checkpoint directory
mkdir -p ~/Desktop/Max\ Planck\ Project/project/checkpoints/hpc_results

# Copy from HPC
scp asahai2024@volta.mpibc.mpg.de:/mnt/beegfs/home/asahai2024/max-planck-project/project/checkpoints/centernet_cem500k_*/detector_best.pt ~/Desktop/Max\ Planck\ Project/project/checkpoints/hpc_results/

scp asahai2024@volta.mpibc.mpg.de:/mnt/beegfs/home/asahai2024/max-planck-project/project/checkpoints/centernet_nature_fast_*/detector_best.pt ~/Desktop/Max\ Planck\ Project/project/checkpoints/hpc_results/
```

### Check Final Logs
```bash
# Get last 20 lines of training logs
ssh asahai2024@volta.mpibc.mpg.de 'tail -20 /mnt/beegfs/home/asahai2024/max-planck-project/project/logs/centernet_*.out'
```

---

## Troubleshooting

### Jobs Don't Start After 5 minutes
```bash
# Check SLURM error
squeue -u asahai2024 -O ReasonList
```

### Jobs Fail with Error
```bash
# Read error log
cat logs/centernet_JOBID.err
```

### GPU Out of Memory
- Job will fail with `RuntimeError: CUDA out of memory`
- Reduce batch_size in SLURM script: 8→4 or 16→8
- Resubmit

### Training Very Slow
- Check: `nvidia-smi` (GPU utilization)
- If <50%: Data loading bottleneck
- Increase: `--num_workers` in SLURM script

---

## Quick Reference Commands

```bash
# Check all your jobs
squeue -u asahai2024

# Check only CenterNet jobs
squeue -u asahai2024 | grep centernet

# Cancel a job (if needed)
scancel JOB_ID

# Check completed jobs
sacct -u asahai2024 -S 2024-03-21

# SSH to HPC
ssh asahai2024@volta.mpibc.mpg.de

# View training output
ssh asahai2024@volta.mpibc.mpg.de 'tail -100 /mnt/beegfs/.../logs/centernet_cem500k_*.out'
```

---

## Expected Results (After ~20 hours)

### CEM500K Baseline
- **Best checkpoint**: `detector_best.pt`
- **Expected F1**: 0.65-0.80
- **Training time**: 18-20 hours
- **Use case**: Baseline comparison

### Nature-Level Accelerated
- **Best checkpoint**: `detector_best.pt`
- **Expected F1**: 0.85-0.92 ✓✓
- **Training time**: 10-14 hours
- **Use case**: Final model for evaluation

---

## Next Steps After Training

1. **Evaluate models**
   ```bash
   python evaluate_detector.py --checkpoint <path> --test_image data/S29/...
   ```

2. **Compare results**
   - F1 scores
   - Precision/Recall
   - 6nm vs 12nm accuracy

3. **If F1 > 0.85**
   - Train ensemble (3 models) for F1 > 0.95
   - Prepare manuscript

4. **If F1 < 0.85**
   - Analyze failure modes
   - Adjust hyperparameters
   - Resubmit

---

## Support Contacts

**HPC Support**: volta.mpibc.mpg.de
**Your Account**: asahai2024
**Project Dir**: /mnt/beegfs/home/asahai2024/max-planck-project/project
**Data**: /mnt/beegfs/.../Max Planck Data/Gold Particle Labelling/analyzed synapses/

---

**Status**: READY FOR SUBMISSION ✅
All systems validated and tested.
