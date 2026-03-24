# Immunogold detection — research & publication protocol

This document defines **what “rigorous” means** for this project: detecting immunogold particles in EM images and **differentiating particle size classes** (e.g. 6 nm vs 12 nm gold), in line with standard reporting in bioimage analysis and particle localization papers.

---

## 1. Scientific tasks (what you claim)

| Task | Definition in this codebase |
|------|-----------------------------|
| **Localization** | Predict \((x, y)\) in **pixel coordinates** for each particle. |
| **Size / type classification** | Assign **class_id** `0` = 6 nm, `1` = 12 nm (see `prepare_labels.py`: `CLASS_TO_ID`). |
| **Joint claim (paper wording)** | “We detect gold particle positions and classify them into size-matched categories evaluated against expert point annotations.” |

Ground truth comes from CSV annotations under each synapse folder (`discover_image_records` / `prepare_labels.py`). Predictions for evaluation are **CSV rows**: `image_id, x, y, class_id, confidence`.

---

## 2. Metrics (primary vs secondary)

### 2.1 Primary (use in abstracts / main tables)

- **Per-class F1** for class 0 (6 nm) and class 1 (12 nm), with precision/recall.
- **Macro-F1** = average of per-class F1s — appropriate when **12 nm is rare** (class imbalance); do **not** report only pooled “micro” F1 unless you also report macro.
- **Mean localization error (px)** for true positives, after greedy matching.

Implementation: `evaluate_detector.py` (per-class matching).

### 2.2 Secondary (recommended for methods / supplement)

- **Threshold sweep**: report **best macro-F1** and the **confidence threshold** used (avoid cherry-picking without stating sweep range). Flags: `--threshold_sweep` or `--sweep_*`.
- **Grouped cross-validation by image**: `--k_folds` — same synapse never appears in both train and test in a fold (avoids leakage).
- **Joint localization + size** (optional flag `--report_joint`):  
  - First match each GT point to the **nearest prediction within radius** (any class).  
  - **Localization recall** = fraction of GT with a match.  
  - **Size accuracy** = among matched pairs, fraction with **correct `class_id`**.  
  - **Confusion matrix** (2×2) on matched pairs — shows **6↔12 swaps** explicitly.

This separates “**did we find the particle?**” from “**did we label the right size?**”, which reviewers often ask for.

### 2.3 What to report alongside numbers

- **`match_dist` (pixels)** and, if possible, **physical distance** (nm) using your effective pixel size — tie to EM scale.
- **Train / val / test split** (which `image_id`s), **random seed**, **checkpoint** path, **inference settings** (tiling, NMS, `max_detections_per_class`, etc.).
- **Failure cases**: empty predictions, threshold saturation, class collapse.

---

## 3. Baselines & comparisons (paper-style)

Compare methods under the **same** `evaluate_detector.py` protocol and same `pred_csv` schema:

| Baseline | Role |
|----------|------|
| LoG / blob + CNN refiner | Classical candidates + learned verifier |
| U-Net heatmap + peaks | Dense baseline |
| CenterNet-style (points + classes) | Direct detection + classification |

Use `benchmark_pipelines.py` where applicable to keep sweeps consistent.

---

## 4. Differentiating size (6 nm vs 12 nm) — operational meaning

- **Per-class evaluation** (`evaluate_detector.py` default): a prediction counts for class `k` only if it is output as class `k` and matches a GT of class `k`. A point predicted as 12 nm near a 6 nm GT **does not** count as a 6 nm TP; it contributes to FP/FN in the usual way. This is **strict**.
- **Joint evaluation** (`--report_joint`): explicitly reports **how often the nearest detection has the wrong class** (confusion matrix on spatially matched pairs). Use this to **quantify size confusion** directly.

For publication, report **both** strict per-class F1 **and** joint size accuracy on localized pairs (they answer different questions).

---

## 5. Reproducibility checklist (before submission)

- [ ] Fixed seeds (Python, NumPy, PyTorch) logged.
- [ ] Exact Slurm / environment: Python version, PyTorch, CUDA, `requirements` or `pip freeze` excerpt.
- [ ] Data path and **no leakage** (image-level splits documented).
- [ ] Checkpoint ID and training command saved (`run_config.json` if using `run_full_detection.py`).
- [ ] Evaluation command line for `evaluate_detector.py` copied into supplement.
- [ ] Optional: bootstrap CIs on macro-F1 across folds (future work if reviewers request).

---

## 6. Commands (reference)

```bash
# Strict per-class + macro-F1 + sweep (example)
python evaluate_detector.py \
  --data_root "/path/to/analyzed synapses" \
  --pred_csv predictions.csv \
  --match_dist 15.0 \
  --k_folds 5 \
  --seed 42 \
  --sweep_start 0.05 --sweep_end 0.5 --sweep_steps 20

# Add joint localization + size confusion report
python evaluate_detector.py \
  --data_root "/path/to/analyzed synapses" \
  --pred_csv predictions.csv \
  --match_dist 15.0 \
  --report_joint
```

---

## 7. Relation to “Nature-level” claims

Peer-reviewed work requires **transparent metrics and controlled comparisons**, not a single headline F1. Treat any **expected F1** from training scripts as a **hypothesis** until measured with `evaluate_detector.py` on a **held-out** image set with the protocol above.

---

*Last updated: aligned with `evaluate_detector.py` + immunogold CSV schema (`class_id` 0/1).*
