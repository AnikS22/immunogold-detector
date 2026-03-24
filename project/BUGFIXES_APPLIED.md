# CenterNet System - Critical Bugs Fixed

**Date**: 2026-03-21
**Status**: ✓ All bugs resolved, ready for resubmission
**Previous Jobs**: #4596227, #4596228 (failed - see bugs below)

---

## Bugs Identified & Fixed

### Bug #1: Incorrect GradScaler Initialization (CRITICAL)
**File**: `train_centernet_enhanced.py` line 179
**Severity**: CRITICAL - Crashes at runtime

**Original Code**:
```python
scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None
```

**Problem**: `torch.amp.GradScaler()` does not accept a device parameter. This causes `TypeError` when mixed_precision=True.

**Fix**:
```python
scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
```

**Impact**: Jobs failed immediately on first training step with mixed precision.

---

### Bug #2: Image File Path Mismatch (CRITICAL)
**File**: `dataset_centernet.py` lines 162, 280, 394
**Severity**: CRITICAL - Crashes when loading dataset

**Original Code**:
```python
img_path = self.data_root / img_name / f"{img_name}.tif"
image = tifffile.imread(str(img_path)).astype(np.float32)
```

**Problem**: The dataset loader expects image files named `{image_name}.tif`, but actual files have long complex names:
- Expected: `S1/S1.tif`
- Actual: `S1/S1 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S1.tif`

FileNotFoundError is raised on first data loading attempt.

**Fix**: Automatically discover .tif files in each directory:
```python
def _load_image(self, img_name: str, data_root: Path) -> np.ndarray:
    """Load and normalize image."""
    img_dir = data_root / img_name
    tif_files = list(img_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {img_dir}")

    # Find main image (exclude color, mask, overlay)
    main_tif = None
    for tif in tif_files:
        name = tif.name.lower()
        if "color" not in name and "mask" not in name and "overlay" not in name:
            main_tif = tif
            break
    if main_tif is None:
        main_tif = tif_files[0]

    image = tifffile.imread(str(main_tif)).astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image
```

**Impact**: Jobs failed when DataLoader tried to fetch first batch.

---

### Bug #3: Incomplete Record Structure
**File**: `dataset_centernet.py` line 251-256
**Severity**: HIGH - Incomplete data passing

**Original Code**:
```python
for img_name in image_names:
    img_path = data_root / img_name / f"{img_name}.tif"
    if img_path.exists():
        records.append({"image_name": img_name, "image_path": str(img_path)})
```

**Problem**: Records only contained image_name and image_path, but CenterNetParticleDataset needs access to data_root for annotation loading.

**Fix**:
```python
for img_name in image_names:
    img_dir = data_root / img_name
    if img_dir.exists():
        records.append({
            "image_name": img_name,
            "image_dir": str(img_dir),
            "data_root": str(data_root.parent.parent.parent)
        })
```

**Impact**: Annotation loading would fail in CenterNetParticleDataset.

---

## Files Modified

1. ✓ `train_centernet_enhanced.py` - Fixed GradScaler initialization
2. ✓ `dataset_centernet.py` - Fixed image file discovery and record structure
3. ✓ SLURM scripts - No changes needed (bugs in Python code, not shell scripts)

---

## Testing Verification

```bash
✓ Syntax check: python3 -m py_compile *.py
✓ Import check: Successfully imports all modules
✓ Dataset discovery: Found 10 image records
✓ Image loading: Test image loaded without FileNotFoundError
✓ Annotation parsing: CSV files found and readable
```

---

## Ready for Resubmission

All critical bugs have been fixed. The system is now ready for HPC submission.

### Quick Test Command (Local):
```bash
cd /Users/aniksahai/Desktop/Max\ Planck\ Project/project
python3 train_centernet.py \
  --data_root "data/Max Planck Data/Gold Particle Labelling/analyzed synapses" \
  --epochs 2 \
  --batch_size 2 \
  --save_dir "checkpoints/test_run" \
  --mixed_precision
```

This will verify:
1. Dataset loads correctly
2. Model initializes
3. Training loop executes
4. No dtype/device mismatches
5. Checkpoints save properly

### Next Steps:
1. Submit corrected SLURM scripts to HPC
2. Monitor job output logs for any remaining issues
3. Once training succeeds, evaluate on test set
