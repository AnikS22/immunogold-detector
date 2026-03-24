# Testing Goals: Immunogold Particle Detection Pipeline

---

## Primary Objective
**Maximize F1 score for detecting 6nm & 12nm gold particles in EM images**

---

## What We're Testing

### ✓ V3 Baseline (Running Now - Job 4594773)
- **Model:** UNetDeepKeypointDetector (4-level, 7.77M params)
- **Config:** pos_weight=30, lr=5e-4, epochs=100
- **Data:** Sliding window (256×256, 50% overlap, 880K views)
- **Expected:** F1 ≈ 0.003-0.010
- **Goal:** Establish baseline performance

### ? V4 Aggressive (If V3 Underwhelms)
- **Config:** pos_weight=200, lr=1e-4, epochs=200
- **Strategy:** Higher recall, accept more false positives
- **Expected:** F1 ≈ 0.005-0.015, Recall 96%
- **Goal:** Test sensitivity trade-off

### ? Alternative Models (Only if V3/V4 Fail)
- **SmallUNetDetector2D:** Lightweight (0.1M), lowest overfit risk
- **NOT GoldDiggerCGAN:** 54M params = massive overfit on 10 images

---

## Key Hypothesis to Test

**Hypothesis:** Reduced sigma (2.5→1.0) + sliding window + realistic augmentations = 30-100× F1 improvement

**Components:**
- ✓ Sigma=1.0 (sharp individual peaks vs overlapping blobs)
- ✓ Sliding window (880× data amplification)
- ✓ 9 EM-realistic augmentations
- ✓ Focal BCE loss (pos_weight for class imbalance)
- ✓ Early stopping (prevent overfit on 10 images)

---

## Success Criteria

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| **F1 Score** | 0.0006 | >0.003 | ? Testing |
| **Precision** | N/A | >0.7 | ? Testing |
| **Recall** | N/A | >0.8 | ? Testing |
| **6nm Detection** | Poor | Good | ? Testing |
| **12nm Detection** | Poor | Good | ? Testing |

---

## Testing Timeline

1. **V3 Baseline** (12-15 hrs)
   - Train UNetDeepKeypointDetector
   - Evaluate on test image
   - Get baseline F1

2. **Decision Point** (after V3 results)
   - If F1 > 0.008: ✓ Good enough
   - If F1 < 0.003: ✗ Try V4 or alternative
   - If F1 0.003-0.008: ? Try V4 aggressive

3. **V4 Aggressive** (if needed, 16-18 hrs)
   - Higher pos_weight, slower learning
   - More epochs, more patience
   - Test precision/recall trade-off

4. **Analysis**
   - Compare V3 vs V4 outputs
   - Optimize detection threshold
   - Validate on held-out test set

---

## Critical Unknowns

- ❓ Will sigma=1.0 fix the massive overlap problem?
- ❓ Is 7.77M parameters too much for 10 images?
- ❓ Can sliding window alone provide enough diversity?
- ❓ Are augmentations realistic enough?

---

## Not Testing (Out of Scope)

- ✗ CGAN (too large for 10 images)
- ✗ 3D models (no volumetric data)
- ✗ Transfer learning (no source data available)
- ✗ Ensemble methods (limited compute budget)

---

## Success Metric: F1 Score Improvement

```
Baseline:        F1 = 0.0006
Target:          F1 = 0.003-0.010  (5-15× improvement)
Ambitious:       F1 = 0.015+       (25×+ improvement)

Current Estimate: 30-100× improvement possible
                  (sigma + sliding window + augmentations)
```

---

## Next Action

**Wait for V3 results (~15 hours from submission)**
- Job ID: 4594773
- Check: `/logs/gold_detector_opt_4594773.out`
- Evaluate: `python run_full_eval.py --model_path checkpoints/4594773/detector_best.pt`
- Decide: V3 good? → Done. V3 bad? → Try V4.
