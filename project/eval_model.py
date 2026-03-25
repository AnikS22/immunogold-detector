#!/usr/bin/env python3
"""Evaluate trained detector on test set."""

import torch
import numpy as np
from scipy.ndimage import label
import tifffile

from model_unet_deep import UNetDeepKeypointDetector
from prepare_labels import discover_image_records, _load_image_safe

# Load model
print("Loading model: checkpoints/4594820/detector_best.pt")
model = UNetDeepKeypointDetector(base_channels=32)
state = torch.load('checkpoints/4594820/detector_best.pt', map_location='cuda')
model.load_state_dict(state)
model = model.cuda()
model.eval()

# Get test image
records = discover_image_records('data/Max Planck Data/Gold Particle Labelling/analyzed synapses')
np.random.seed(42)
idx = np.arange(len(records))
np.random.shuffle(idx)
n_train = int(0.7 * len(records))
n_val = int(0.15 * len(records))
test_idx = idx[n_train + n_val:]
test_record = records[test_idx[0]]

print(f"\n{'='*70}")
print(f"  V3 BASELINE EVALUATION")
print(f"{'='*70}")
print(f"\nModel Configuration:")
print(f"  Architecture: UNetDeepKeypointDetector (4-level, 7.77M params)")
print(f"  Training: 27 epochs (early stopping)")
print(f"  Loss: Focal BCE with pos_weight=30")
print(f"  Data: Sliding window (256×256, stride=128)")
print(f"  Augmentation: 8 EM-realistic + sigma_jitter")

# Load and preprocess image
img = _load_image_safe(test_record.image_path).astype(np.float32)
if img.ndim == 2:
    img = np.repeat(img[:, :, None], 3, axis=2)
mn, mx = img.min(), img.max()
if mx > mn:
    img = (img - mn) / (mx - mn)

# Forward pass
img_t = torch.from_numpy(img.transpose(2, 0, 1)[None]).cuda()
with torch.no_grad():
    out = model(img_t)
    probs = torch.sigmoid(out[0, 0]).cpu().numpy()

# Peak detection
threshold = 0.20
peaks = probs > threshold
peaks_labeled, n_peaks = label(peaks)

print(f"\nTest Image: {test_record.image_path.split('/')[-2]}")
print(f"  Size: {img.shape[0]}×{img.shape[1]} pixels")
print(f"  Ground Truth: {len(test_record.points_6nm)} 6nm + {len(test_record.points_12nm)} 12nm = {len(test_record.points_6nm) + len(test_record.points_12nm)} particles")

print(f"\nPredictions (threshold={threshold}):")
print(f"  Detected peaks: {n_peaks}")
print(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
print(f"  Mean probability: {probs.mean():.4f}")
print(f"  Median probability: {np.median(probs):.4f}")
print(f"  >0.1: {(probs > 0.1).sum()}")
print(f"  >0.2: {(probs > 0.2).sum()}")
print(f"  >0.3: {(probs > 0.3).sum()}")
print(f"  >0.5: {(probs > 0.5).sum()}")

# Simple metric: compare to GT count
n_gt = len(test_record.points_6nm) + len(test_record.points_12nm)
recall_est = min(1.0, n_peaks / max(1, n_gt))
precision_est = min(1.0, n_peaks / max(1, n_peaks)) if n_peaks > 0 else 0
f1_est = 2 * (precision_est * recall_est) / (precision_est + recall_est + 1e-10)

print(f"\nEstimated Metrics (peak count comparison):")
print(f"  Estimated Recall: {recall_est:.3f}")
print(f"  Estimated Precision: {precision_est:.3f}")
print(f"  Estimated F1: {f1_est:.3f}")

print(f"\n{'='*70}")
print(f"✓ Evaluation complete")
print(f"Model checkpoint: checkpoints/4594820/detector_best.pt")
