import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from prepare_labels import ID_TO_CLASS, discover_image_records


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    mean_localization_error: float


def load_predictions(path: str) -> Dict[str, Dict[int, List[Tuple[float, float, float]]]]:
    out: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(lambda: {0: [], 1: []})
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            x = float(row["x"])
            y = float(row["y"])
            cls = int(row["class_id"])
            conf = float(row["confidence"])
            out[image_id][cls].append((x, y, conf))
    return out


def greedy_match(
    gt: np.ndarray, pred: List[Tuple[float, float, float]], max_dist: float
) -> Tuple[int, int, int, List[float]]:
    """Match predictions to ground truth using Hungarian algorithm (optimal assignment)."""
    if len(gt) == 0:
        return 0, len(pred), 0, []
    pred_xy = np.array([[p[0], p[1]] for p in pred], dtype=np.float32) if pred else np.zeros((0, 2), np.float32)
    if len(pred_xy) == 0:
        return 0, 0, len(gt), []

    dist_matrix = np.sqrt(
        np.sum((gt[:, np.newaxis, :] - pred_xy[np.newaxis, :, :]) ** 2, axis=2)
    )
    cost = dist_matrix.copy()
    cost[cost > max_dist] = max_dist * 100

    gt_indices, pred_indices = linear_sum_assignment(cost)

    tp = 0
    dists: List[float] = []
    matched_pred = set()
    for gi, pi in zip(gt_indices, pred_indices):
        d = dist_matrix[gi, pi]
        if d <= max_dist:
            tp += 1
            dists.append(float(d))
            matched_pred.add(pi)

    fn = int(len(gt) - tp)
    fp = int(len(pred_xy) - len(matched_pred))
    return tp, fp, fn, dists


def calc_metrics(tp: int, fp: int, fn: int, loc_errors: Sequence[float]) -> Metrics:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    mean_loc = float(np.mean(loc_errors)) if len(loc_errors) > 0 else float("nan")
    return Metrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        mean_localization_error=mean_loc,
    )


def filter_predictions_by_threshold(
    pred_map: Dict[str, Dict[int, List[Tuple[float, float, float]]]], threshold: float
) -> Dict[str, Dict[int, List[Tuple[float, float, float]]]]:
    out: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(lambda: {0: [], 1: []})
    for image_id, cls_map in pred_map.items():
        for cls in [0, 1]:
            out[image_id][cls] = [p for p in cls_map.get(cls, []) if p[2] >= threshold]
    return out


def evaluate_subset(
    gt_map: Dict[str, Dict[int, np.ndarray]],
    pred_map: Dict[str, Dict[int, List[Tuple[float, float, float]]]],
    match_dist: float,
    image_ids: Sequence[str],
) -> Dict[str, Metrics]:
    per_cls = {
        0: {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []},
        1: {"tp": 0, "fp": 0, "fn": 0, "loc_errors": []},
    }
    total_tp = total_fp = total_fn = 0
    loc_errors_all: List[float] = []

    for image_id in image_ids:
        points = gt_map[image_id]
        preds = pred_map.get(image_id, {0: [], 1: []})
        for cls in [0, 1]:
            tp, fp, fn, d = greedy_match(points[cls], preds.get(cls, []), max_dist=match_dist)
            per_cls[cls]["tp"] += tp
            per_cls[cls]["fp"] += fp
            per_cls[cls]["fn"] += fn
            per_cls[cls]["loc_errors"].extend(d)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            loc_errors_all.extend(d)

    m_all = calc_metrics(total_tp, total_fp, total_fn, loc_errors_all)
    m_6 = calc_metrics(
        int(per_cls[0]["tp"]),
        int(per_cls[0]["fp"]),
        int(per_cls[0]["fn"]),
        per_cls[0]["loc_errors"],
    )
    m_12 = calc_metrics(
        int(per_cls[1]["tp"]),
        int(per_cls[1]["fp"]),
        int(per_cls[1]["fn"]),
        per_cls[1]["loc_errors"],
    )
    macro_f1 = 0.5 * (m_6.f1 + m_12.f1)
    return {
        "all": m_all,
        "6nm": m_6,
        "12nm": m_12,
        "macro": Metrics(
            precision=float("nan"),
            recall=float("nan"),
            f1=float(macro_f1),
            tp=0,
            fp=0,
            fn=0,
            mean_localization_error=float("nan"),
        ),
    }


def joint_match_image(
    gt_map: Dict[int, np.ndarray],
    pred_map_cls: Dict[int, List[Tuple[float, float, float]]],
    match_dist: float,
) -> Tuple[int, int, int, int, int, List[float], np.ndarray]:
    """
    Pool all GT points with class labels and all predictions with class labels.
    For each GT (in class order 0 then 1), assign nearest unmatched prediction within match_dist.

    Returns:
        tp_loc: localized matches (any class)
        tp_cls: matches with correct class_id
        wrong_cls: localized but wrong class
        fp: unmatched predictions
        fn: unmatched GT
        loc_errors: distances for localized pairs
        confusion: 2x2 counts [gt_row][pred_col] for matched pairs only
    """
    gt_list: List[Tuple[float, float, int]] = []
    for cls in [0, 1]:
        arr = gt_map.get(cls, np.zeros((0, 2)))
        for i in range(len(arr)):
            gt_list.append((float(arr[i, 0]), float(arr[i, 1]), cls))

    pred_list: List[Tuple[float, float, float, int]] = []
    for cls in [0, 1]:
        for p in pred_map_cls.get(cls, []):
            pred_list.append((float(p[0]), float(p[1]), float(p[2]), cls))

    if len(gt_list) == 0:
        return 0, 0, 0, len(pred_list), 0, [], np.zeros((2, 2), dtype=np.int64)

    pred_xy = np.array([[p[0], p[1]] for p in pred_list], dtype=np.float32) if pred_list else np.zeros((0, 2), np.float32)
    used = np.zeros(len(pred_list), dtype=bool)
    loc_errors: List[float] = []
    confusion = np.zeros((2, 2), dtype=np.int64)
    tp_loc = tp_cls = wrong_cls = 0

    for gx, gy, gcls in gt_list:
        if len(pred_xy) == 0:
            continue
        dist = np.sqrt(((pred_xy - np.array([[gx, gy]], dtype=np.float32)) ** 2).sum(axis=1))
        dist[used] = 1e9
        j = int(np.argmin(dist))
        if dist[j] < match_dist:
            used[j] = True
            tp_loc += 1
            loc_errors.append(float(dist[j]))
            pcls = pred_list[j][3]
            confusion[gcls, pcls] += 1
            if pcls == gcls:
                tp_cls += 1
            else:
                wrong_cls += 1

    fp = int((~used).sum())
    fn = len(gt_list) - tp_loc
    return tp_loc, tp_cls, wrong_cls, fp, fn, loc_errors, confusion


def evaluate_joint(
    gt_map: Dict[str, Dict[int, np.ndarray]],
    pred_map: Dict[str, Dict[int, List[Tuple[float, float, float]]]],
    match_dist: float,
    image_ids: Sequence[str],
) -> Dict[str, object]:
    """Pooled joint localization + size classification across images."""
    total_tp_loc = total_tp_cls = total_wrong = total_fp = total_fn = 0
    loc_all: List[float] = []
    confusion = np.zeros((2, 2), dtype=np.int64)

    for image_id in image_ids:
        points = gt_map[image_id]
        preds = pred_map.get(image_id, {0: [], 1: []})
        tp_loc, tp_cls, wrong_cls, fp, fn, loc_err, cm = joint_match_image(points, preds, match_dist)
        total_tp_loc += tp_loc
        total_tp_cls += tp_cls
        total_wrong += wrong_cls
        total_fp += fp
        total_fn += fn
        loc_all.extend(loc_err)
        confusion += cm

    # Localization treated as binary detection (each GT is one event)
    prec_loc = total_tp_loc / max(1, total_tp_loc + total_fp)
    rec_loc = total_tp_loc / max(1, total_tp_loc + total_fn)
    f1_loc = 2 * prec_loc * rec_loc / max(1e-8, prec_loc + rec_loc)
    mean_loc = float(np.mean(loc_all)) if loc_all else float("nan")

    size_acc = float(total_tp_cls / max(1, total_tp_loc)) if total_tp_loc > 0 else float("nan")

    return {
        "localization": Metrics(
            precision=float(prec_loc),
            recall=float(rec_loc),
            f1=float(f1_loc),
            tp=int(total_tp_loc),
            fp=int(total_fp),
            fn=int(total_fn),
            mean_localization_error=mean_loc,
        ),
        "size_accuracy_on_matched": size_acc,
        "n_matched": int(total_tp_loc),
        "n_wrong_class_among_matched": int(total_wrong),
        "confusion_matched_pairs": confusion,
    }


def build_grouped_folds(image_ids: Sequence[str], k_folds: int, seed: int) -> List[List[str]]:
    if k_folds < 2:
        return [list(image_ids)]
    ids = np.array(sorted(image_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds: List[List[str]] = [[] for _ in range(k_folds)]
    for i, image_id in enumerate(ids.tolist()):
        folds[i % k_folds].append(image_id)
    return folds


def parse_thresholds(
    threshold: float,
    threshold_sweep: str,
    sweep_start: float,
    sweep_end: float,
    sweep_steps: int,
) -> List[float]:
    if threshold_sweep.strip():
        vals = [float(s.strip()) for s in threshold_sweep.split(",") if s.strip()]
        return sorted(set(vals))
    if sweep_steps > 1:
        vals = np.linspace(sweep_start, sweep_end, int(sweep_steps)).tolist()
        return sorted(set(float(v) for v in vals))
    return [float(threshold)]


def print_metrics_block(name: str, m: Metrics) -> None:
    print(f"{name}.precision={m.precision:.4f}")
    print(f"{name}.recall={m.recall:.4f}")
    print(f"{name}.f1={m.f1:.4f}")
    print(f"{name}.mean_localization_error={m.mean_localization_error:.4f}")
    print(f"{name}.tp={m.tp} {name}.fp={m.fp} {name}.fn={m.fn}")


def print_joint_block(j: Dict[str, object]) -> None:
    loc = j["localization"]  # type: ignore[assignment]
    assert isinstance(loc, Metrics)
    cm = j["confusion_matched_pairs"]  # type: ignore[assignment]
    assert isinstance(cm, np.ndarray)
    print("--- joint (localize first, then score size on matched pairs) ---")
    print_metrics_block("joint.localization", loc)
    print(f"joint.size_accuracy_on_matched={j['size_accuracy_on_matched']:.4f}")
    print(f"joint.n_matched={j['n_matched']} joint.n_wrong_class_among_matched={j['n_wrong_class_among_matched']}")
    print("joint.confusion_rows_GT_cols_PRED (matched pairs only):")
    for r in [0, 1]:
        row = ID_TO_CLASS[r]
        print(
            f"  GT {row}: pred {ID_TO_CLASS[0]}={cm[r,0]} pred {ID_TO_CLASS[1]}={cm[r,1]}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate keypoint detections against GT coordinates.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--pred_csv", type=str, required=True)
    p.add_argument("--match_dist", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=0.0, help="Minimum prediction confidence.")
    p.add_argument(
        "--threshold_sweep",
        type=str,
        default="",
        help="Comma-separated confidence thresholds (e.g. 0.01,0.02,0.03).",
    )
    p.add_argument("--sweep_start", type=float, default=0.0)
    p.add_argument("--sweep_end", type=float, default=0.6)
    p.add_argument("--sweep_steps", type=int, default=0)
    p.add_argument("--k_folds", type=int, default=1, help="Grouped K-fold by image_id.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report_joint",
        action="store_true",
        help="Report joint localization + size accuracy and 2x2 confusion on spatially matched pairs.",
    )
    args = p.parse_args()

    records = discover_image_records(args.data_root)
    gt_map = {r.image_id: r.points for r in records}
    pred_map_raw = load_predictions(args.pred_csv)
    image_ids = sorted(gt_map.keys())
    folds = build_grouped_folds(image_ids, args.k_folds, args.seed)
    thresholds = parse_thresholds(
        threshold=args.threshold,
        threshold_sweep=args.threshold_sweep,
        sweep_start=args.sweep_start,
        sweep_end=args.sweep_end,
        sweep_steps=args.sweep_steps,
    )

    best_row = None
    best_macro = -1.0
    for thr in thresholds:
        pred_map = filter_predictions_by_threshold(pred_map_raw, threshold=thr)
        if len(folds) == 1:
            metrics = evaluate_subset(gt_map, pred_map, args.match_dist, image_ids)
            print(f"threshold={thr:.6f}")
            print_metrics_block("all", metrics["all"])
            print_metrics_block("class_6nm", metrics["6nm"])
            print_metrics_block("class_12nm", metrics["12nm"])
            print(f"macro.f1={metrics['macro'].f1:.4f}")
            macro_f1 = metrics["macro"].f1
            if args.report_joint:
                joint = evaluate_joint(gt_map, pred_map, args.match_dist, image_ids)
                print_joint_block(joint)
        else:
            fold_macros: List[float] = []
            fold_f1_all: List[float] = []
            fold_joint_size_acc: List[float] = []
            for i, fold_ids in enumerate(folds):
                metrics = evaluate_subset(gt_map, pred_map, args.match_dist, fold_ids)
                fold_macros.append(metrics["macro"].f1)
                fold_f1_all.append(metrics["all"].f1)
                print(
                    f"threshold={thr:.6f} fold={i + 1}/{len(folds)} "
                    f"n_images={len(fold_ids)} all_f1={metrics['all'].f1:.4f} "
                    f"macro_f1={metrics['macro'].f1:.4f}"
                )
                if args.report_joint:
                    joint = evaluate_joint(gt_map, pred_map, args.match_dist, fold_ids)
                    print(f"  fold {i + 1} joint.size_accuracy_on_matched={joint['size_accuracy_on_matched']:.4f}")
                    fold_joint_size_acc.append(float(joint["size_accuracy_on_matched"]))  # type: ignore[arg-type]
            macro_f1 = float(np.mean(fold_macros))
            print(
                f"threshold={thr:.6f} grouped_cv_mean_all_f1={float(np.mean(fold_f1_all)):.4f} "
                f"grouped_cv_mean_macro_f1={macro_f1:.4f}"
            )
            if args.report_joint and fold_joint_size_acc:
                print(
                    f"threshold={thr:.6f} grouped_cv_mean_joint_size_accuracy="
                    f"{float(np.nanmean(fold_joint_size_acc)):.4f}"
                )

        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_row = (thr, macro_f1)

    if best_row is not None and len(thresholds) > 1:
        print(f"best_threshold={best_row[0]:.6f} best_macro_f1={best_row[1]:.4f}")


if __name__ == "__main__":
    main()
