import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from prepare_labels import discover_image_records


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    mean_localization_error: float


def load_predictions(path: str) -> Dict[str, List[Tuple[float, float, float]]]:
    out: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["image_id"]].append((float(row["x"]), float(row["y"]), float(row["confidence"])))
    return out


def filter_by_threshold(pred_map: Dict[str, List[Tuple[float, float, float]]], threshold: float):
    return {k: [p for p in v if p[2] >= threshold] for k, v in pred_map.items()}


def greedy_match(gt: np.ndarray, pred: List[Tuple[float, float, float]], max_dist: float):
    if len(gt) == 0:
        return 0, len(pred), 0, []
    pred_xy = np.array([[p[0], p[1]] for p in pred], dtype=np.float32) if pred else np.zeros((0, 2), np.float32)
    used = np.zeros(len(pred_xy), dtype=bool)
    tp = 0
    dists: List[float] = []
    for g in gt:
        if len(pred_xy) == 0:
            continue
        dist = np.sqrt(((pred_xy - g[None, :]) ** 2).sum(axis=1))
        dist[used] = 1e9
        j = int(np.argmin(dist))
        if dist[j] < max_dist:
            used[j] = True
            tp += 1
            dists.append(float(dist[j]))
    fp = int((~used).sum())
    fn = int(len(gt) - tp)
    return tp, fp, fn, dists


def calc_metrics(tp: int, fp: int, fn: int, loc_errors: Sequence[float]) -> Metrics:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return Metrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        mean_localization_error=float(np.mean(loc_errors)) if len(loc_errors) > 0 else float("nan"),
    )


def parse_thresholds(args: argparse.Namespace) -> List[float]:
    if args.threshold_sweep.strip():
        vals = [float(s.strip()) for s in args.threshold_sweep.split(",") if s.strip()]
        return sorted(set(vals))
    if args.sweep_steps > 1:
        return sorted(set(float(v) for v in np.linspace(args.sweep_start, args.sweep_end, int(args.sweep_steps)).tolist()))
    return [float(args.threshold)]


def main() -> None:
    p = argparse.ArgumentParser(description="Binary (all-particles) evaluation.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--pred_csv", type=str, required=True)
    p.add_argument("--match_dist", type=float, default=15.0)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--threshold_sweep", type=str, default="")
    p.add_argument("--sweep_start", type=float, default=0.0)
    p.add_argument("--sweep_end", type=float, default=0.6)
    p.add_argument("--sweep_steps", type=int, default=0)
    args = p.parse_args()

    records = discover_image_records(args.data_root)
    gt_map = {r.image_id: np.concatenate([r.points[0], r.points[1]], axis=0) for r in records}
    pred_map_raw = load_predictions(args.pred_csv)
    image_ids = sorted(gt_map.keys())
    thresholds = parse_thresholds(args)

    best = (-1.0, None)
    for thr in thresholds:
        pred_map = filter_by_threshold(pred_map_raw, thr)
        tp = fp = fn = 0
        loc: List[float] = []
        for image_id in image_ids:
            a, b, c, d = greedy_match(gt_map[image_id], pred_map.get(image_id, []), max_dist=args.match_dist)
            tp += a
            fp += b
            fn += c
            loc.extend(d)
        m = calc_metrics(tp, fp, fn, loc)
        print(f"threshold={thr:.6f}")
        print(f"binary.precision={m.precision:.4f}")
        print(f"binary.recall={m.recall:.4f}")
        print(f"binary.f1={m.f1:.4f}")
        print(f"binary.mean_localization_error={m.mean_localization_error:.4f}")
        print(f"binary.tp={m.tp} binary.fp={m.fp} binary.fn={m.fn}")
        if m.f1 > best[0]:
            best = (m.f1, thr)
    if best[1] is not None and len(thresholds) > 1:
        print(f"best_threshold={best[1]:.6f} best_binary_f1={best[0]:.4f}")


if __name__ == "__main__":
    main()
