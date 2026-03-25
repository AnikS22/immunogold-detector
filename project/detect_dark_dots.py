"""
Immunogold detector — find the darkest dots in EM images.

Gold particles are simply the darkest, most isolated dark pixels.
No blob detection, no LoG, no complexity. Just:
  1. Find very dark pixels
  2. Group them into clusters
  3. Take cluster centroids as particle locations
  4. Classify by cluster size (6nm = small, 12nm = larger)

Usage:
  python detect_dark_dots.py --data_root "path/to/analyzed synapses" --visualize
"""

import argparse
import csv
import os
import sys
from typing import List, Tuple

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import label as ndimage_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_labels import discover_image_records, _load_image_safe


def find_dark_dots(
    image: np.ndarray,
    dark_percentile: float = 2.0,
    max_cluster_size: int = 150,
    min_cluster_size: int = 1,
) -> List[Tuple[float, float, float, float]]:
    """
    Find dark dot clusters in an EM image.

    Returns: list of (x, y, area, mean_intensity)
    """
    # Normalize to [0, 1]
    img = image.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return []
    img = (img - mn) / (mx - mn)

    # Threshold: keep only the darkest pixels
    thresh = np.percentile(img, dark_percentile)
    dark_mask = img <= thresh

    # Label connected components
    labeled, n_comp = ndimage_label(dark_mask)
    if n_comp == 0:
        return []

    detections = []
    for comp_id in range(1, n_comp + 1):
        component = (labeled == comp_id)
        area = int(component.sum())

        if area < min_cluster_size or area > max_cluster_size:
            continue

        ys, xs = np.where(component)
        cx = float(xs.mean())
        cy = float(ys.mean())
        mean_val = float(img[component].mean())

        detections.append((cx, cy, float(area), mean_val))

    return detections


def classify_by_area(
    detections: List[Tuple[float, float, float, float]],
    area_boundary: float = 8.0,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """6nm = small clusters, 12nm = larger clusters."""
    class_6nm = []
    class_12nm = []
    for x, y, area, intensity in detections:
        conf = 1.0 - intensity  # darker = more confident
        if area <= area_boundary:
            class_6nm.append((x, y, conf))
        else:
            class_12nm.append((x, y, conf))
    return class_6nm, class_12nm


def greedy_match(gt, pred, max_dist):
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
    return tp, int((~used).sum()), int(len(gt) - tp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="dark_dots_results")
    p.add_argument("--match_dist", type=float, default=5.0)
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = discover_image_records(args.data_root)
    print(f"Found {len(records)} images\n")

    if args.visualize:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    # Sweep percentile and area boundary
    print("=== Sweep ===")
    best_f1 = 0
    best_params = (2.0, 8.0, 150)

    for pct in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]:
        for max_sz in [20, 50, 100, 150]:
            for area_bnd in [5.0, 8.0, 12.0, 20.0]:
                s_tp, s_fp, s_fn = 0, 0, 0
                for rec in records:
                    img = _load_image_safe(rec.image_path)
                    if img.ndim == 3:
                        img = img.mean(axis=2)

                    dets = find_dark_dots(img, dark_percentile=pct, max_cluster_size=max_sz)
                    d6, d12 = classify_by_area(dets, area_boundary=area_bnd)

                    tp6, fp6, fn6 = greedy_match(rec.points[0], d6, args.match_dist)
                    tp12, fp12, fn12 = greedy_match(rec.points[1], d12, args.match_dist)
                    s_tp += tp6 + tp12
                    s_fp += fp6 + fp12
                    s_fn += fn6 + fn12

                prec = s_tp / max(1, s_tp + s_fp)
                rec_v = s_tp / max(1, s_tp + s_fn)
                f1 = 2 * prec * rec_v / max(1e-8, prec + rec_v)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (pct, area_bnd, max_sz)
                    print(f"  NEW BEST: pct={pct} area_bnd={area_bnd} max_sz={max_sz} | "
                          f"tp={s_tp} fp={s_fp} fn={s_fn} P={prec:.4f} R={rec_v:.4f} F1={f1:.4f}")

    pct, area_bnd, max_sz = best_params
    print(f"\nBest: pct={pct} area_bnd={area_bnd} max_sz={max_sz} F1={best_f1:.4f}")

    # Final run with best params
    print(f"\n=== Final Run ===\n")
    total_tp = {0: 0, 1: 0}
    total_fp = {0: 0, 1: 0}
    total_fn = {0: 0, 1: 0}

    for rec in records:
        img = _load_image_safe(rec.image_path)
        if img.ndim == 3:
            img = img.mean(axis=2)

        dets = find_dark_dots(img, dark_percentile=pct, max_cluster_size=max_sz)
        d6, d12 = classify_by_area(dets, area_boundary=area_bnd)

        tp6, fp6, fn6 = greedy_match(rec.points[0], d6, args.match_dist)
        tp12, fp12, fn12 = greedy_match(rec.points[1], d12, args.match_dist)
        total_tp[0] += tp6; total_fp[0] += fp6; total_fn[0] += fn6
        total_tp[1] += tp12; total_fp[1] += fp12; total_fn[1] += fn12

        print(f"  {rec.image_id}: det={len(dets)} gt={len(rec.points[0])+len(rec.points[1])} "
              f"6nm(tp={tp6} fp={fp6} fn={fn6}) 12nm(tp={tp12} fp={fp12} fn={fn12})")

        if args.visualize:
            mn, mx = float(img.min()), float(img.max())
            disp = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            axes[0].imshow(disp, cmap="gray")
            if len(rec.points[0]) > 0:
                axes[0].scatter(rec.points[0][:, 0], rec.points[0][:, 1],
                               s=50, facecolors="none", edgecolors="lime", linewidths=1, label="GT 6nm")
            if len(rec.points[1]) > 0:
                axes[0].scatter(rec.points[1][:, 0], rec.points[1][:, 1],
                               s=80, facecolors="none", edgecolors="yellow", linewidths=1, label="GT 12nm")
            axes[0].legend(fontsize=9)
            axes[0].set_title(f"{rec.image_id} — Ground Truth")

            axes[1].imshow(disp, cmap="gray")
            if d6:
                axes[1].scatter([d[0] for d in d6], [d[1] for d in d6],
                               s=20, c="cyan", marker="+", linewidths=0.8, label=f"Det 6nm ({len(d6)})")
            if d12:
                axes[1].scatter([d[0] for d in d12], [d[1] for d in d12],
                               s=30, c="magenta", marker="+", linewidths=0.8, label=f"Det 12nm ({len(d12)})")
            if len(rec.points[0]) > 0:
                axes[1].scatter(rec.points[0][:, 0], rec.points[0][:, 1],
                               s=50, facecolors="none", edgecolors="lime", linewidths=0.5, alpha=0.4)
            if len(rec.points[1]) > 0:
                axes[1].scatter(rec.points[1][:, 0], rec.points[1][:, 1],
                               s=80, facecolors="none", edgecolors="yellow", linewidths=0.5, alpha=0.4)
            axes[1].legend(fontsize=9)
            axes[1].set_title(f"Detections (TP={tp6+tp12} FP={fp6+fp12} FN={fn6+fn12})")

            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"{rec.image_id}.png"), dpi=150)
            plt.close()

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for cls_name, cls_id in [("6nm", 0), ("12nm", 1)]:
        tp, fp, fn = total_tp[cls_id], total_fp[cls_id], total_fn[cls_id]
        prec = tp / max(1, tp + fp)
        rec_v = tp / max(1, tp + fn)
        f1 = 2 * prec * rec_v / max(1e-8, prec + rec_v)
        print(f"  {cls_name}: tp={tp} fp={fp} fn={fn} P={prec:.4f} R={rec_v:.4f} F1={f1:.4f}")

    all_tp = total_tp[0] + total_tp[1]
    all_fp = total_fp[0] + total_fp[1]
    all_fn = total_fn[0] + total_fn[1]
    all_prec = all_tp / max(1, all_tp + all_fp)
    all_rec = all_tp / max(1, all_tp + all_fn)
    all_f1 = 2 * all_prec * all_rec / max(1e-8, all_prec + all_rec)
    print(f"\n  ALL: tp={all_tp} fp={all_fp} fn={all_fn} P={all_prec:.4f} R={all_rec:.4f} F1={all_f1:.4f}")


if __name__ == "__main__":
    main()
