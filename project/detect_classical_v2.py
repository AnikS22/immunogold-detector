"""
Classical immunogold detector v2 — intensity + circularity filtering.

Gold particles are:
  1. Very dark (lowest intensity in the image)
  2. Small and circular
  3. High local contrast against surrounding tissue

This version uses intensity thresholding + LoG + circularity filtering
to dramatically reduce false positives.

Usage:
  python detect_classical_v2.py --data_root "path/to/analyzed synapses" --visualize
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import label as ndimage_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_labels import discover_image_records, _load_image_safe


def detect_dark_particles(
    image: np.ndarray,
    intensity_percentile: float = 5.0,
    min_size: int = 2,
    max_size: int = 80,
    circularity_thresh: float = 0.4,
    local_contrast_sigma: float = 10.0,
    local_contrast_thresh: float = 0.08,
) -> List[Tuple[float, float, float, float]]:
    """
    Detect dark circular particles using intensity + shape filtering.

    Returns list of (x, y, equivalent_radius, confidence).
    """
    h, w = image.shape

    # Step 1: Normalize image to [0, 1]
    mn, mx = float(image.min()), float(image.max())
    if mx > mn:
        img = (image - mn) / (mx - mn)
    else:
        return []

    # Step 2: Local contrast — how much darker is each pixel than its neighborhood
    local_mean = ndimage.gaussian_filter(img, sigma=local_contrast_sigma)
    darkness = local_mean - img  # positive where pixel is darker than surroundings

    # Step 3: Threshold — keep only pixels that are both:
    #   a) Dark in absolute terms (below percentile)
    #   b) Dark relative to local neighborhood
    intensity_thresh = np.percentile(img, intensity_percentile)
    dark_mask = (img < intensity_thresh) & (darkness > local_contrast_thresh)

    # Step 4: Clean up with morphological operations
    # Close small gaps, then open to remove thin structures
    struct = ndimage.generate_binary_structure(2, 1)
    dark_mask = ndimage.binary_closing(dark_mask, structure=struct, iterations=1)
    dark_mask = ndimage.binary_opening(dark_mask, structure=struct, iterations=1)

    # Step 5: Connected components
    labeled, n_components = ndimage_label(dark_mask)

    if n_components == 0:
        return []

    detections = []

    for comp_id in range(1, n_components + 1):
        component = (labeled == comp_id)
        area = int(component.sum())

        # Size filter
        if area < min_size or area > max_size:
            continue

        # Bounding box
        ys, xs = np.where(component)
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1

        # Circularity: area / (pi * (max_dim/2)^2)
        # Perfect circle = 1.0, elongated = low
        max_dim = max(bbox_h, bbox_w)
        if max_dim == 0:
            continue
        circularity = area / (np.pi * (max_dim / 2) ** 2)

        if circularity < circularity_thresh:
            continue

        # Centroid
        cy = float(ys.mean())
        cx = float(xs.mean())

        # Equivalent radius
        eq_radius = np.sqrt(area / np.pi)

        # Confidence: combination of darkness and circularity
        mean_darkness = float(darkness[component].mean())
        mean_intensity = float(img[component].mean())
        confidence = mean_darkness * circularity * (1.0 - mean_intensity)

        detections.append((cx, cy, eq_radius, confidence))

    return detections


def classify_by_radius(
    detections: List[Tuple[float, float, float, float]],
    radius_boundary: float = 3.0,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Split by equivalent radius into 6nm (small) and 12nm (large)."""
    class_6nm = []
    class_12nm = []
    for x, y, radius, conf in detections:
        if radius <= radius_boundary:
            class_6nm.append((x, y, conf))
        else:
            class_12nm.append((x, y, conf))
    return class_6nm, class_12nm


def greedy_match(
    gt: np.ndarray, pred: List[Tuple[float, float, float]], max_dist: float
) -> Tuple[int, int, int]:
    if len(gt) == 0:
        return 0, len(pred), 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    pred_xy = np.array([[p[0], p[1]] for p in pred], dtype=np.float32)
    used = np.zeros(len(pred_xy), dtype=bool)
    tp = 0
    for g in gt:
        dist = np.sqrt(((pred_xy - g[None, :]) ** 2).sum(axis=1))
        dist[used] = 1e9
        j = int(np.argmin(dist))
        if dist[j] < max_dist:
            used[j] = True
            tp += 1
    fp = int((~used).sum())
    fn = int(len(gt) - tp)
    return tp, fp, fn


def main():
    p = argparse.ArgumentParser(description="Classical immunogold detector v2")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="classical_v2_results")
    p.add_argument("--match_dist", type=float, default=5.0)
    p.add_argument("--visualize", action="store_true")
    # Tunable params
    p.add_argument("--intensity_percentile", type=float, default=3.0)
    p.add_argument("--local_contrast_thresh", type=float, default=0.05)
    p.add_argument("--circularity_thresh", type=float, default=0.4)
    p.add_argument("--min_size", type=int, default=2)
    p.add_argument("--max_size", type=int, default=80)
    p.add_argument("--radius_boundary", type=float, default=3.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = discover_image_records(args.data_root)
    print(f"Found {len(records)} images\n")

    if args.visualize:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    # ================================================================
    # PARAMETER SWEEP
    # ================================================================
    print("=== Parameter Sweep ===")
    sweep_configs = [
        (1.0, 0.03, 0.3),
        (2.0, 0.03, 0.3),
        (3.0, 0.05, 0.4),
        (3.0, 0.05, 0.5),
        (5.0, 0.05, 0.4),
        (5.0, 0.08, 0.5),
        (5.0, 0.08, 0.6),
        (7.0, 0.10, 0.5),
        (10.0, 0.10, 0.5),
        (3.0, 0.03, 0.3),
        (3.0, 0.08, 0.5),
        (2.0, 0.05, 0.5),
    ]

    best_f1 = 0
    best_config = sweep_configs[0]

    for pct, lc_thresh, circ_thresh in sweep_configs:
        sweep_tp, sweep_fp, sweep_fn = 0, 0, 0
        for rec in records:
            img = _load_image_safe(rec.image_path)
            if img.ndim == 3:
                img = img.mean(axis=2)
            img = img.astype(np.float32)

            dets = detect_dark_particles(
                img,
                intensity_percentile=pct,
                local_contrast_thresh=lc_thresh,
                circularity_thresh=circ_thresh,
                min_size=args.min_size,
                max_size=args.max_size,
            )
            det_6nm, det_12nm = classify_by_radius(dets, args.radius_boundary)

            tp6, fp6, fn6 = greedy_match(rec.points[0], det_6nm, args.match_dist)
            tp12, fp12, fn12 = greedy_match(rec.points[1], det_12nm, args.match_dist)
            sweep_tp += tp6 + tp12
            sweep_fp += fp6 + fp12
            sweep_fn += fn6 + fn12

        prec = sweep_tp / max(1, sweep_tp + sweep_fp)
        rec_val = sweep_tp / max(1, sweep_tp + sweep_fn)
        f1 = 2 * prec * rec_val / max(1e-8, prec + rec_val)
        total_det = sweep_tp + sweep_fp
        print(f"  pct={pct:.1f} lc={lc_thresh:.2f} circ={circ_thresh:.1f} | "
              f"dets={total_det:6d} tp={sweep_tp:3d} fp={sweep_fp:5d} fn={sweep_fn:3d} "
              f"P={prec:.4f} R={rec_val:.4f} F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_config = (pct, lc_thresh, circ_thresh)

    pct, lc_thresh, circ_thresh = best_config
    print(f"\nBest: pct={pct} lc={lc_thresh} circ={circ_thresh} F1={best_f1:.4f}")

    # ================================================================
    # RUN WITH BEST CONFIG
    # ================================================================
    print(f"\n=== Final Run ===\n")

    total_tp = {0: 0, 1: 0}
    total_fp = {0: 0, 1: 0}
    total_fn = {0: 0, 1: 0}
    csv_rows = [["image_id", "x", "y", "class_id", "confidence", "radius"]]

    for rec in records:
        img = _load_image_safe(rec.image_path)
        if img.ndim == 3:
            img = img.mean(axis=2)
        img = img.astype(np.float32)

        dets = detect_dark_particles(
            img,
            intensity_percentile=pct,
            local_contrast_thresh=lc_thresh,
            circularity_thresh=circ_thresh,
            min_size=args.min_size,
            max_size=args.max_size,
        )
        det_6nm, det_12nm = classify_by_radius(dets, args.radius_boundary)

        tp6, fp6, fn6 = greedy_match(rec.points[0], det_6nm, args.match_dist)
        tp12, fp12, fn12 = greedy_match(rec.points[1], det_12nm, args.match_dist)
        total_tp[0] += tp6; total_fp[0] += fp6; total_fn[0] += fn6
        total_tp[1] += tp12; total_fp[1] += fp12; total_fn[1] += fn12

        print(f"  {rec.image_id}: det={len(dets)} gt={len(rec.points[0])+len(rec.points[1])} "
              f"6nm(tp={tp6} fp={fp6} fn={fn6}) 12nm(tp={tp12} fp={fp12} fn={fn12})")

        for x, y, conf in det_6nm:
            rad = next((d[2] for d in dets if d[0] == x and d[1] == y), 0)
            csv_rows.append([rec.image_id, f"{x:.2f}", f"{y:.2f}", "0", f"{conf:.4f}", f"{rad:.2f}"])
        for x, y, conf in det_12nm:
            rad = next((d[2] for d in dets if d[0] == x and d[1] == y), 0)
            csv_rows.append([rec.image_id, f"{x:.2f}", f"{y:.2f}", "1", f"{conf:.4f}", f"{rad:.2f}"])

        if args.visualize:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Normalize for display
            mn, mx = float(img.min()), float(img.max())
            disp = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Raw with GT
            axes[0].imshow(disp, cmap="gray")
            if len(rec.points[0]) > 0:
                axes[0].scatter(rec.points[0][:, 0], rec.points[0][:, 1],
                               s=50, facecolors="none", edgecolors="lime", linewidths=1, label="GT 6nm")
            if len(rec.points[1]) > 0:
                axes[0].scatter(rec.points[1][:, 0], rec.points[1][:, 1],
                               s=80, facecolors="none", edgecolors="yellow", linewidths=1, label="GT 12nm")
            axes[0].legend(fontsize=9)
            axes[0].set_title(f"{rec.image_id} — Ground Truth")

            # Raw with detections
            axes[1].imshow(disp, cmap="gray")
            if det_6nm:
                axes[1].scatter([d[0] for d in det_6nm], [d[1] for d in det_6nm],
                               s=20, c="cyan", marker="+", linewidths=0.8, label=f"Det 6nm ({len(det_6nm)})")
            if det_12nm:
                axes[1].scatter([d[0] for d in det_12nm], [d[1] for d in det_12nm],
                               s=30, c="magenta", marker="+", linewidths=0.8, label=f"Det 12nm ({len(det_12nm)})")
            # Also show GT as circles for comparison
            if len(rec.points[0]) > 0:
                axes[1].scatter(rec.points[0][:, 0], rec.points[0][:, 1],
                               s=50, facecolors="none", edgecolors="lime", linewidths=0.5, alpha=0.5)
            if len(rec.points[1]) > 0:
                axes[1].scatter(rec.points[1][:, 0], rec.points[1][:, 1],
                               s=80, facecolors="none", edgecolors="yellow", linewidths=0.5, alpha=0.5)
            axes[1].legend(fontsize=9)
            tp_total = tp6 + tp12
            fp_total = fp6 + fp12
            fn_total = fn6 + fn12
            axes[1].set_title(f"Detections (TP={tp_total} FP={fp_total} FN={fn_total})")

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"{rec.image_id}_v2.png"), dpi=150)
            plt.close()

    # ================================================================
    # FINAL METRICS
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for cls_name, cls_id in [("6nm", 0), ("12nm", 1)]:
        tp = total_tp[cls_id]
        fp = total_fp[cls_id]
        fn = total_fn[cls_id]
        prec = tp / max(1, tp + fp)
        rec_val = tp / max(1, tp + fn)
        f1 = 2 * prec * rec_val / max(1e-8, prec + rec_val)
        print(f"  {cls_name}: tp={tp} fp={fp} fn={fn} P={prec:.4f} R={rec_val:.4f} F1={f1:.4f}")

    all_tp = total_tp[0] + total_tp[1]
    all_fp = total_fp[0] + total_fp[1]
    all_fn = total_fn[0] + total_fn[1]
    all_prec = all_tp / max(1, all_tp + all_fp)
    all_rec = all_tp / max(1, all_tp + all_fn)
    all_f1 = 2 * all_prec * all_rec / max(1e-8, all_prec + all_rec)

    print(f"\n  ALL:  tp={all_tp} fp={all_fp} fn={all_fn} P={all_prec:.4f} R={all_rec:.4f} F1={all_f1:.4f}")

    csv_path = os.path.join(args.out_dir, "predictions_v2.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()
