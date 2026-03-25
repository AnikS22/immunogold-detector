#!/usr/bin/env python3
"""Compute F1 score on test image."""

import torch
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist

from model_unet_deep import UNetDeepKeypointDetector
from prepare_labels import discover_image_records, _load_image_safe
import tifffile

def peak_detection(heatmap, threshold=0.20, min_distance=10):
    """Find peaks in heatmap."""
    peaks = heatmap > threshold
    if not peaks.any():
        return np.array([]).reshape(0, 2)

    labeled, n_peaks = label(peaks)
    coords = center_of_mass(peaks, labeled, range(1, n_peaks + 1))
    if len(coords) == 0:
        return np.array([]).reshape(0, 2)

    coords = np.array(coords)  # (N, 2) in (y, x) format

    # Simple NMS: keep peaks with enough distance
    if len(coords) > 1:
        kept = [0]
        for i in range(1, len(coords)):
            dists = np.linalg.norm(coords[kept] - coords[i], axis=1)
            if (dists > min_distance).all():
                kept.append(i)
        coords = coords[kept]

    return coords

def compute_metrics(pred_coords, gt_coords, match_dist=15.0):
    """Compute TP, FP, FN and metrics."""
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return {'TP': 0, 'FP': 0, 'FN': 0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    if len(pred_coords) == 0:
        return {'TP': 0, 'FP': 0, 'FN': len(gt_coords), 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    if len(gt_coords) == 0:
        return {'TP': 0, 'FP': len(pred_coords), 'FN': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Compute distances between predictions and ground truth
    distances = cdist(pred_coords, gt_coords)

    # Match predictions to GT (greedy assignment)
    matched_pred = set()
    matched_gt = set()
    tp = 0

    for gt_idx in range(len(gt_coords)):
        best_pred_idx = np.argmin(distances[:, gt_idx])
        best_dist = distances[best_pred_idx, gt_idx]
        if best_dist <= match_dist and best_pred_idx not in matched_pred:
            tp += 1
            matched_pred.add(best_pred_idx)
            matched_gt.add(gt_idx)

    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Load model
print("Loading model...")
model = UNetDeepKeypointDetector(base_channels=32)
state = torch.load('checkpoints/4594820/detector_best.pt', map_location='cpu')
model.load_state_dict(state)
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

print(f"Test image: {test_record.image_path.split('/')[-2]}")

# Load image
img = _load_image_safe(test_record.image_path).astype(np.float32)
if img.ndim == 2:
    img = np.repeat(img[:, :, None], 3, axis=2)
mn, mx = img.min(), img.max()
if mx > mn:
    img = (img - mn) / (mx - mn)

# Inference
print("Running inference...")
img_t = torch.from_numpy(img.transpose(2, 0, 1)[None]).cpu()
with torch.no_grad():
    out = model(img_t)
    probs = torch.sigmoid(out[0, 0]).numpy()

# Peak detection with threshold 0.20 (V3 conservative)
threshold = 0.20
print(f"Detecting peaks (threshold={threshold})...")
pred_coords = peak_detection(probs, threshold=threshold)  # (N, 2) in (y, x)
print(f"  Predicted peaks: {len(pred_coords)}")

# Get ground truth (convert from (x, y) to (y, x))
gt_6nm = np.array(test_record.points_6nm)  # (x, y)
gt_12nm = np.array(test_record.points_12nm)  # (x, y)
if len(gt_6nm) > 0:
    gt_6nm = gt_6nm[:, [1, 0]]  # Convert to (y, x)
if len(gt_12nm) > 0:
    gt_12nm = gt_12nm[:, [1, 0]]  # Convert to (y, x)
gt_coords = np.vstack([gt_6nm, gt_12nm]) if (len(gt_6nm) > 0 or len(gt_12nm) > 0) else np.array([]).reshape(0, 2)
print(f"  Ground truth: {len(gt_6nm)} 6nm + {len(gt_12nm)} 12nm = {len(gt_coords)} total")

# Compute metrics
metrics = compute_metrics(pred_coords, gt_coords, match_dist=15.0)

print(f"\n{'='*70}")
print(f"  V3 BASELINE RESULTS - JOB 4594820")
print(f"{'='*70}")
print(f"\nModel Configuration:")
print(f"  Architecture: UNetDeepKeypointDetector (7.77M params)")
print(f"  Training: 27 epochs, early stopping")
print(f"  Loss: Focal BCE with pos_weight=30")
print(f"  Learning rate: 5e-4 with cosine annealing")
print(f"  Data: Sliding window (256×256, stride=128, ~880K views)")
print(f"  Augmentation: 8 EM-realistic + sigma_jitter")
print(f"  Sigma: 1.0 (optimized from 2.5)")

print(f"\nTest Image: {test_record.image_path.split('/')[-2]}")
print(f"  Size: 2048×2048")
print(f"  Ground truth: {len(gt_coords)} particles")

print(f"\nPrediction Statistics:")
print(f"  Detection threshold: {threshold}")
print(f"  Predicted peaks: {len(pred_coords)}")
print(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
print(f"  Mean probability: {probs.mean():.4f}")

print(f"\nDetection Metrics (match_dist=15px):")
print(f"  TP (True Positives): {metrics['TP']}")
print(f"  FP (False Positives): {metrics['FP']}")
print(f"  FN (False Negatives): {metrics['FN']}")
print(f"")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  F1 Score: {metrics['f1']:.4f}")

print(f"\n{'='*70}")
if metrics['f1'] > 0.003:
    print(f"✓ F1 IMPROVED: {metrics['f1']:.4f} (target was >0.003)")
else:
    print(f"⚠ F1 below target: {metrics['f1']:.4f}")
print(f"{'='*70}\n")
