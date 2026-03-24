# CenterNet Job Monitoring Setup

**Submission Time**: 2026-03-21 (Ready)
**Expected Duration**: ~14-20 hours (both jobs in parallel)

---

## Jobs to Monitor

### Job 1: CEM500K Baseline
- **Script**: `hpc/train_centernet_cem500k.slurm`
- **Expected Duration**: 18-20 hours
- **Resources**: 8 CPUs, 48GB RAM, 1 GPU
- **Expected F1**: 0.65-0.80
- **Output**: `checkpoints/centernet_cem500k_JOBID/detector_best.pt`

### Job 2: Nature-Level Accelerated
- **Script**: `hpc/train_centernet_nature_fast.slurm`
- **Expected Duration**: 10-14 hours
- **Resources**: 16 CPUs, 96GB RAM, 1 GPU
- **Expected F1**: 0.85-0.92
- **Output**: `checkpoints/centernet_nature_fast_JOBID/detector_best.pt`

---

## Monitoring Commands

### Check Job Status (Real-time)
```bash
squeue -u asahai2024 | grep centernet
```

### Check Specific Jobs
```bash
squeue -u asahai2024 -j JOB_ID1,JOB_ID2 --format="%.18i %.9P %.25j %.8u %.2t %.10M %.6D %N"
```

### Monitor Training Progress (Live)
```bash
# For Job 1:
tail -f /mnt/beegfs/home/asahai2024/max-planck-project/project/logs/centernet_cem500k_*.out

# For Job 2:
tail -f /mnt/beegfs/home/asahai2024/max-planck-project/project/logs/centernet_nature_fast_*.out
```

### Check GPU Usage
```bash
srun -u asahai2024 nvidia-smi
```

### List Completed Jobs
```bash
sacct -u asahai2024 --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## What to Monitor

### Good Signs (Training Proceeding Normally)
✓ Job status: `RUNNING`
✓ Loss decreasing over epochs
✓ No CUDA out of memory errors
✓ GPU utilization 80-95%
✓ Training speed: ~0.5-1.0 sec/batch

### Warning Signs
⚠ Job status: `PENDING` for >30 min (resource contention)
⚠ Out of memory (OOM) error
⚠ NaN or Inf in loss values
⚠ GPU utilization <50% (data loading bottleneck)
⚠ Training extremely slow (<0.1 sec/batch)

### Critical Issues
❌ Job status: `FAILED` or `TIMEOUT`
❌ Python errors or import failures
❌ Data loading errors
❌ GPU crash

---

## Expected Timeline

```
Time 0:00   - Jobs submitted
Time 0:05   - Jobs start initializing (RUNNING state)
Time 0:10   - Dependencies installing (pip install)
Time 0:20   - Python training script starts
Time 0:30   - First epoch completes
Time 2:00   - ~30 minutes training, loss should be declining
Time 6:00   - ~25% complete
Time 10:00  - Job 2 (fast) may complete first
Time 14:00  - Job 2 should finish (10-14h window)
Time 18:00  - Job 1 should be near completion
Time 20:00  - Both jobs finished
```

---

## After Jobs Complete

### 1. Check Exit Status
```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
# Should show: State=COMPLETED, ExitCode=0:0
```

### 2. Retrieve Best Models
```bash
# Copy checkpoint files from HPC
scp asahai2024@volta.mpibc.mpg.de:/mnt/beegfs/.../checkpoints/centernet_cem500k_*/detector_best.pt ./checkpoints/
scp asahai2024@volta.mpibc.mpg.de:/mnt/beegfs/.../checkpoints/centernet_nature_fast_*/detector_best.pt ./checkpoints/
```

### 3. Check Training Logs
```bash
# View final loss values
tail -20 logs/centernet_cem500k_*.out
tail -20 logs/centernet_nature_fast_*.out
```

### 4. Evaluate Results
```bash
python evaluate_detector.py \
  --checkpoint ./checkpoints/centernet_cem500k_JOBID/detector_best.pt \
  --test_image data/S29/...
```

### 5. Compare Performance
```bash
# Generate comparison metrics
python compare_models.py \
  --model1 ./checkpoints/centernet_cem500k_JOBID/detector_best.pt \
  --model2 ./checkpoints/centernet_nature_fast_JOBID/detector_best.pt \
  --test_set data/...
```

---

## Quick Reference: Job IDs

Fill in after submission:

```
Job 1 (CEM500K):      #_______
Job 2 (Nature-fast):  #_______

Submitted: ________________
Expected completion: ________________
```

---

## Troubleshooting

### If Job Fails to Start
```bash
# Check SLURM error log
cat /mnt/beegfs/.../logs/centernet_cem500k_JOBID.err
```

### If Job Runs Out of Memory
- Reduce batch_size: 8 → 4
- Reduce patch_size: 256 → 128
- Disable mixed_precision

### If Training is Very Slow
- Check GPU utilization: `nvidia-smi`
- Increase num_workers: 7 → 12
- Check disk I/O: `iostat`

### If Loss is NaN
- Reduce learning rate: 1e-4 → 1e-5
- Check for data loading issues
- Verify gradient accumulation isn't overflowing

---

## Expected Output

After successful training, you'll have:

```
checkpoints/
├── centernet_cem500k_JOBID/
│   ├── detector_best.pt          # Best model (validation)
│   └── checkpoints_final.pt      # Final model (epoch 100)
└── centernet_nature_fast_JOBID/
    ├── detector_best.pt
    └── checkpoints_final.pt

logs/
├── centernet_cem500k_JOBID.out   # Training logs
├── centernet_cem500k_JOBID.err   # Error logs
├── centernet_nature_fast_JOBID.out
└── centernet_nature_fast_JOBID.err
```

---

## Next Steps After Monitoring

1. **If both jobs succeed (F1 > 0.80)**:
   - Use nature-level model for final predictions
   - Train ensemble (3 models) for F1 > 0.95
   - Prepare manuscript

2. **If only CEM500K succeeds (F1 < 0.65)**:
   - Analyze why nature-level failed
   - Adjust hyperparameters
   - Resubmit

3. **If both fail**:
   - Check error logs
   - Verify data integrity
   - Debug locally first
   - Resubmit with fixes
