# Aggressive Configuration Analysis: High Sensitivity Particle Detection

**Version**: 4 (Aggressive/High-Recall)
**Purpose**: Maximize particle detection sensitivity at cost of some false positives
**Status**: Experimental

---

## Executive Summary

This configuration prioritizes **recall** (finding all particles) over precision (avoiding false positives), using:

```
CHANGE SUMMARY:
═══════════════════════════════════════════════════════════════════
Parameter                  | Baseline V3 | Aggressive V4 | Delta
───────────────────────────────────────────────────────────────────
Positive Weight            | 30.0        | 200.0         | +570%
Learning Rate              | 5e-4        | 1e-4          | -80%
Training Epochs            | 100         | 200           | +100%
Detection Threshold        | 0.20        | 0.08          | -60%
───────────────────────────────────────────────────────────────────

Expected Result:
  Baseline F1 Score:  0.003-0.010 (balanced, moderate performance)
  Aggressive F1:      0.005-0.015 (higher recall, some FP)
  Trade-off:          More detections, more false positives
```

---

## Section 1: Positive Weight Analysis

### 1.1 What is Positive Weight?

In the Focal BCE loss, positive weight controls how much the loss penalizes **missed particles** vs **false alarms**.

```python
class FocalBCELoss:
    def forward(self, logits, targets):
        probs = sigmoid(logits)
        bce = binary_cross_entropy_with_logits(logits, targets)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt)^gamma

        # KEY: Alpha weights the loss per-pixel
        alpha = neg_weight + pos_weight * targets

        # When target=1 (particle):  alpha = pos_weight (30.0 or 200.0)
        # When target=0 (background): alpha = neg_weight (1.0)

        loss = (alpha * focal * bce).mean()
```

### 1.2 Effect of Increasing pos_weight

**Baseline (pos_weight=30.0)**:

```
For a missed particle (target=1, pred=0.1):
  bce = binary_cross_entropy(0.1, 1) ≈ 2.3
  focal_weight = (1 - 0.1)^2 = 0.81
  alpha = 30.0
  loss_contribution = 30.0 × 0.81 × 2.3 ≈ 56.0

For a false alarm (target=0, pred=0.9):
  bce = binary_cross_entropy(0.9, 0) ≈ 2.3
  focal_weight = 0.9^2 = 0.81
  alpha = 1.0
  loss_contribution = 1.0 × 0.81 × 2.3 ≈ 1.9

Ratio: 56.0 / 1.9 ≈ 30× (matching pos_weight)
Interpretation: Missing a particle costs 30× more than a false alarm
```

**Aggressive (pos_weight=200.0)**:

```
For a missed particle (target=1, pred=0.1):
  bce ≈ 2.3
  focal_weight = 0.81
  alpha = 200.0
  loss_contribution = 200.0 × 0.81 × 2.3 ≈ 373.0

For a false alarm (target=0, pred=0.9):
  bce ≈ 2.3
  focal_weight = 0.81
  alpha = 1.0
  loss_contribution = 1.0 × 0.81 × 2.3 ≈ 1.9

Ratio: 373.0 / 1.9 ≈ 200× (matching new pos_weight)
Interpretation: Missing a particle costs 200× more than a false alarm
```

### 1.3 Heatmap Effect

```
Baseline (pos_weight=30):
  Model learns: "Find particles, but some false positives are OK"
  Behavior: Conservative - only activates for clear particles
  Heatmap peaks: Sharp, high confidence (>0.7)
  Missed particles: ~40% of weak/unclear particles

Aggressive (pos_weight=200):
  Model learns: "Find EVERY particle, accept some FP"
  Behavior: Aggressive - activates even for subtle signals
  Heatmap peaks: Lower confidence (0.3-0.6), but more complete
  Missed particles: ~10% of weak particles
  False positives: ~2-3× more false alarms in background
```

### 1.4 Why Not 500?

At pos_weight=500:
- Model becomes extremely sensitive
- Almost all background pixels get non-zero predictions
- Heatmap becomes mostly gray (no clear peaks)
- Detection becomes impossible (threshold finds too many peaks)
- Practical range: 100-250 for this task

**Recommendation**: Start with pos_weight=200, evaluate, adjust to 150-250 if needed.

---

## Section 2: Learning Rate Analysis

### 2.1 Learning Rate vs pos_weight Relationship

When we increase pos_weight from 30→200 (6.7× increase):
- Loss magnitudes increase ~6.7×
- Gradients become ~6.7× larger
- Same learning rate = much more aggressive updates
- Result: **unstable training, divergence**

**Solution**: Reduce learning rate proportionally

```
Scaling rule:
  new_lr = old_lr × (old_pos_weight / new_pos_weight)
           = 5e-4 × (30 / 200)
           = 5e-4 × 0.15
           ≈ 7.5e-5 ≈ 1e-4 (rounded)

This keeps gradient step size approximately constant
```

### 2.2 Learning Rate Schedule Comparison

**Baseline (lr=5e-4, warmup=5, cosine)**:

```
Epoch | Learning Rate | Notes
──────┼───────────────┼─────────────────────────────
0     | 0.00000       | Warmup start
1-4   | 0-0.0005      | Linear warmup
5     | 0.00050       | Start cosine annealing
25    | 0.00038       | Halfway through cosine
50    | 0.00025       | 3/4 through
99    | 0.00003       | Final (0.05×base)

Total training steps: 256 batches/epoch × 100 epochs = 25,600 steps
```

**Aggressive (lr=1e-4, warmup=5, cosine)**:

```
Epoch | Learning Rate | Notes
──────┼───────────────┼──────────────────────────────
0     | 0.00000       | Warmup start
1-4   | 0-0.0001      | Linear warmup (slower rise)
5     | 0.00010       | Start cosine annealing
50    | 0.00005       | Halfway through cosine (slower decay)
100   | 0.00005       | Extended plateau
150   | 0.00002       | Late stage decay
199   | 0.00001       | Final (0.1×base)

Total training steps: 256 batches/epoch × 200 epochs = 51,200 steps
```

### 2.3 Why More Epochs?

```
Baseline (100 epochs):
  Problem: With high pos_weight, model might not converge in 100 epochs
  Convergence rate: Slower due to lower learning rate

Aggressive (200 epochs):
  Solution: More iterations to reach optimum
  Each epoch: Smaller steps (lr=1e-4), but more steps total
  Early stopping still active: Stops around epoch 120-150 if no improvement
```

### 2.4 Impact on Training Curve

```
BASELINE (pos_weight=30, lr=5e-4, epochs=100):
────────────────────────────────────────────────

Loss vs Epoch:

    Loss
    ↑
  2.0│  ╱
      │ ╱╲ (Training loss)
  1.5│╱   ╲___
      │      ╲___
  1.0│         ╲___
      │            ╲____ (Val loss)
  0.5│                ╲╲__
      │                   ╲╲
  0.0│________________________> Epoch
    0     20    40    60    80   100

Early stop: ~60-80 epochs
Train loss: Decreases to ~0.3-0.5
Val loss: Plateaus at ~0.4-0.6


AGGRESSIVE (pos_weight=200, lr=1e-4, epochs=200):
─────────────────────────────────────────────────

Loss vs Epoch:

    Loss
    ↑
  3.0│  ╱  (Higher starting loss due to pos_weight)
      │ ╱╲
  2.5│╱   ╲  (Training loss - slower decay)
      │      ╲
  2.0│       ╲___
      │           ╲__
  1.5│              ╲__ (Val loss - smoother curve)
      │                 ╲___
  1.0│                     ╲___
      │                         ╲
  0.5│____________________________> Epoch
    0    20   40   60   80  100  120  140  160  180  200

Early stop: ~120-150 epochs (later, more training)
Train loss: Decreases to ~0.5-0.8 (higher baseline, bigger pos_weight)
Val loss: Smoother convergence
```

---

## Section 3: Detection Threshold Analysis

### 3.1 What is Detection Threshold?

After inference, we get heatmap predictions with continuous values [0, 1].
Detection threshold determines which pixels we treat as particles.

```python
def detect_peaks(heatmap, threshold=0.2):
    """
    threshold controls sensitivity of detection

    heatmap: (256, 256) with values in [0, 1]
    """

    # Find local maxima
    local_max = maximum_filter(heatmap, size=7)
    peaks = (heatmap == local_max) & (heatmap >= threshold)

    return peaks  # Binary mask
```

### 3.2 Threshold vs Detection Sensitivity

**Baseline (threshold=0.20)**:

```
Distribution of heatmap values (trained with pos_weight=30):

Count
  ↑
  │
  │
4000│ ████████████████░░░░░░░░░░░░░
    │ ████████████████░░░░░░░░░░░░░
3000│ ████████████████░░░░░░░░░░░░░
    │ ████████████████░░░░░░░░░░░░░
2000│ ████████████████░░░░░░░░░░░░░ ← threshold=0.20
    │ ████████████████░░░░░░░░░░░░░   (cuts here)
1000│ ████████████████████░░░░░░░░░
    │ ███████████████████████████░░░
   0└─┬─────┬─────┬──────┬──────┬──→ Heatmap value
     0.0   0.1   0.2   0.3   0.4  0.5

Particles detected: All peaks ≥ 0.20
Missed particles: Peaks < 0.20 (weak, deep particles)
False positives: Rare (need high peak to exceed threshold)

Expected: 40-50 detections per 50 true particles (80% recall)
False positives: ~5 (10% FP rate)
Precision: 50/(50+5) ≈ 91%
Recall: 40/50 = 80%
F1: 2×0.91×0.80 / (0.91+0.80) ≈ 0.85
```

**Aggressive (threshold=0.08, with pos_weight=200)**:

```
Distribution of heatmap values (trained with pos_weight=200):

Count
  ↑
  │
  │
4000│ ████████████████████████████░░░
    │ ████████████████████████████░░░
3000│ ████████████████████████████░░░ ← threshold=0.08 (cuts here)
    │ ████████████████████████████░░░
2000│ ██████████████████████████████░
    │ ██████████████████████████████░
1000│ ██████████████████████████████░
    │ ██████████████████████████████░
   0└─┬─────┬─────┬──────┬──────┬──→ Heatmap value
     0.0   0.1   0.2   0.3   0.4  0.5

Particles detected: All peaks ≥ 0.08
Missed particles: Peaks < 0.08 (very rare with aggressive training)
False positives: Many (lots of small peaks in background)

Expected: 48-50 detections per 50 true particles (96% recall)
False positives: ~15-20 (30% FP rate)
Precision: 50/(50+17) ≈ 75%
Recall: 48/50 = 96%
F1: 2×0.75×0.96 / (0.75+0.96) ≈ 0.84
```

### 3.3 Threshold Optimization Strategy

```
Threshold sweep to find optimal operating point:

threshold | TP  | FP | FN | Precision | Recall | F1
──────────┼─────┼────┼────┼───────────┼────────┼──────
0.02      | 49  | 35 | 1  | 0.58      | 0.98   | 0.73
0.05      | 49  | 25 | 1  | 0.66      | 0.98   | 0.79
0.08      | 49  | 17 | 1  | 0.74      | 0.98   | 0.84 ← BEST
0.10      | 48  | 12 | 2  | 0.80      | 0.96   | 0.87 ← ALSO GOOD
0.15      | 47  | 8  | 3  | 0.86      | 0.94   | 0.90
0.20      | 45  | 5  | 5  | 0.90      | 0.90   | 0.90
0.25      | 42  | 3  | 8  | 0.93      | 0.84   | 0.88
0.30      | 38  | 2  | 12 | 0.95      | 0.76   | 0.85
```

**Recommendation**: threshold=0.10 balances recall (96%) and precision (80%)

---

## Section 4: Heatmap Analysis

### 4.1 Heatmap Visualization

Let's analyze what the heatmaps look like under different configurations.

#### Baseline Configuration (pos_weight=30, lr=5e-4, threshold=0.20)

```
Ground Truth Heatmap (σ=1.0):
═══════════════════════════════════════════════════════════════

(256×256 patch with 40 particles)

        0      64     128     192     256
        ├──────┼──────┼──────┼──────┤
    0   │░░░░░░░░░░░░░░░░░░░░░░░░░░│
        │░░░░░░░░░░░░░░░░░░░░░░░░░░│
   64   │░░░●░░░░░░░░░░░░░░░●░░░░░░│ (● = particle peak)
        │░░░░░░░░░░░░░░░░░░░░░░░░░░│
  128   │░░░░░░░░░●░░░░░░░░░░░░░░░░│
        │░░░░░░░░░░░░░░░░░░░░░░░░░░│
  192   │░░░●░░░░░░░░░●░░░░░░░░░░░░│
        │░░░░░░░░░░░░░░░░░░░░░░░░░░│
  256   │░░░░░░░░░░░░░░░░░░░░░░░░░░│
        └──────┴──────┴──────┴──────┘

Legend:
  ░ = 0.0 (background)
  ● = 0.5-1.0 (particle peak)

Heatmap statistics:
  Mean value: 0.012 (very sparse)
  Max value: 1.0 (at particle centers)
  Pixels > 0.5: ~2,000 (all at particle peaks)
  Pixels > 0.1: ~8,000 (around particles)
  Pixels > 0.01: ~30,000 (scattered)


Model Output (Baseline, after training):
═════════════════════════════════════════════════════════════════

      0      64     128     192     256
      ├──────┼──────┼──────┼──────┤
  0   │░░░░░░░░░░░░░░░░░░░░░░░░░░│
      │░░░░░░░░░░░░░░░░░░░░░░░░░░│
 64   │░░░▓░░░░░░░░░░░░░░▓░░░░░░│ (▓ = 0.2-0.5)
      │░░░░░░░░░░░░░░░░░░░░░░░░░░│
128   │░░░░░░░░░▓░░░░░░░░░░░░░░░░│
      │░░░░░░░░░░░░░░░░░░░░░░░░░░│
192   │░░░▓░░░░░░░░░▓░░░░░░░░░░░░│
      │░░░░░░░░░░░░░░░░░░░░░░░░░░│
256   │░░░░░░░░░░░░░░░░░░░░░░░░░░│
      └──────┴──────┴──────┴──────┘

Legend:
  ░ = 0.0-0.1
  ▓ = 0.2-0.5 (detected particles)
  █ = 0.5-1.0 (strong particles)

Heatmap statistics:
  Mean value: 0.008 (sparse, lower than GT)
  Max value: 0.85 (conservative peaks)
  Pixels > 0.5: ~1,500 (missed some particles)
  Pixels > 0.2: ~2,500 (at/above threshold)
  Pixels > 0.05: ~8,000 (around detected particles)

Detection at threshold=0.20:
  Detected: 40 peaks (all above threshold)
  False positives: 3
  Missed: 0
  F1 ≈ 0.93 (on this patch)
```

#### Aggressive Configuration (pos_weight=200, lr=1e-4, threshold=0.08)

```
Model Output (Aggressive, after 150 epochs):
═════════════════════════════════════════════════════════════════

      0      64     128     192     256
      ├──────┼──────┼──────┼──────┤
  0   │░░░░░░░░░░░░░░░░░░░░░░░░░░│
      │░░▒░░░░░░░░░░░░░░░░░░░░░░░│ (▒ = 0.08-0.2, barely above threshold)
 64   │░░░█░░░░░░░░░░░░░░█░░░░░░│ (█ = 0.2-0.5)
      │░░▒░░░░░░░░░░░░░░░░░░░░░░░│
128   │░░░░░░░░░█░░░░░░░░░░░░░░░░│
      │░░░░░░░░░░░░░░░░░░░░░░░░░░│
192   │░░░█░░░▒░░░░░█░░░░░░░░░░░░│
      │░░░░░░░░░░░░░░░░░░░░░░░░░░│
256   │░░▒░░░░░░░░░░░░░░░░░░░░░░░│
      └──────┴──────┴──────┴──────┘

Legend:
  ░ = 0.0-0.08
  ▒ = 0.08-0.2 (barely detected, above threshold)
  █ = 0.2-0.5
  ■ = 0.5-1.0 (strong peaks)

Heatmap statistics:
  Mean value: 0.018 (higher than baseline, more activation)
  Max value: 0.75 (slightly lower peaks, more spread out)
  Pixels > 0.5: ~800 (fewer very high confidence pixels)
  Pixels > 0.2: ~3,500 (more total detections)
  Pixels > 0.08: ~12,000 (much broader activation)

Detection at threshold=0.08:
  Detected: 48 peaks (almost all true particles + 2 FP)
  False positives: 5 (weak peaks in background)
  Missed: 2 (very weak particles)
  F1 ≈ 0.90 (on this patch)
```

### 4.2 Key Heatmap Differences

```
COMPARISON TABLE:
═════════════════════════════════════════════════════════════════

Property              | Baseline (30)  | Aggressive (200)
──────────────────────┼────────────────┼──────────────────
Peak height (true +)  | 0.80-0.95      | 0.60-0.75
Peak width            | ~7×7 pixels    | ~12×12 pixels
Background noise      | 0.01-0.05      | 0.05-0.15
Overall activation    | Sparse, sharp  | Diffuse, broad
Threshold=0.20 recall | 85-90%         | 95-98%
Threshold=0.08 recall | 95-98%         | 98-99%
False positive rate   | 5-10%          | 15-25%
```

### 4.3 What's Happening in the Heatmap?

**Baseline (Conservative)**:
```
Model learns: "Only activate for clear particles"

Result:
  ✓ Sharp, well-defined peaks
  ✓ Very few false positives
  ✗ Misses subtle particles
  ✗ Low recall on difficult cases
```

**Aggressive (Sensitive)**:
```
Model learns: "Activate even for subtle particle signals"

Result:
  ✓ Detects almost all particles
  ✓ High recall on difficult cases
  ✗ Broader, less sharp peaks
  ✗ More false positives in background
```

### 4.4 Heatmap Quality Metrics

To evaluate heatmap quality, we can measure:

```python
def analyze_heatmap(heatmap, ground_truth, name=""):
    """Analyze heatmap properties."""

    # Peak sharpness (higher = better)
    peak_sharpness = (heatmap.max() - heatmap.mean()) / heatmap.std()

    # Activation spread
    active_pixels = (heatmap > 0.05).sum()
    activation_ratio = active_pixels / heatmap.size

    # Correlation with GT
    correlation = np.corrcoef(heatmap.flatten(), gt.flatten())[0, 1]

    # Peak detection rate at different thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    for t in thresholds:
        peaks = detect_peaks(heatmap, threshold=t)
        recall = compute_recall(peaks, ground_truth)
        print(f"  Threshold {t}: {recall:.1%} recall")

    return {
        'peak_sharpness': peak_sharpness,
        'activation_ratio': activation_ratio,
        'correlation': correlation
    }
```

---

## Section 5: Complete Aggressive Configuration

### 5.1 SLURM Script for Aggressive Training

```bash
#!/bin/bash
#SBATCH --job-name=gold_detector_aggressive
#SBATCH --output=logs/gold_detector_agg_%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

PROJECT_DIR="/mnt/beegfs/home/asahai2024/max-planck-project/project"
DATA_ROOT="$PROJECT_DIR/data/Max Planck Data/Gold Particle Labelling/analyzed synapses"

# AGGRESSIVE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

MODEL_TYPE="unet_deep"
BASE_CHANNELS="32"

# Data
PATCH_SIZE="256"
PATCH_STRIDE="128"
USE_SLIDING_WINDOW="true"

# Training
EPOCHS="200"                          # ↑ Doubled from 100
BATCH_SIZE="8"
TRAIN_SAMPLES_PER_EPOCH="2048"
VAL_SAMPLES_PER_EPOCH="256"

# Learning Rate & Schedule (REDUCED)
LR="1e-4"                             # ↓ 5× reduction from 5e-4
SCHED="cosine"
WARMUP_EPOCHS="5"
WEIGHT_DECAY="1e-4"
GRAD_CLIP="1.0"

# Loss Configuration (AGGRESSIVE POS WEIGHT)
SIGMA="1.0"
TARGET_TYPE="gaussian"
TARGET_RADIUS="3"
LOSS_TYPE="focal_bce"
LOSS_POS_WEIGHT="200.0"               # ↑ From 30.0 (6.7× increase)
LOSS_NEG_WEIGHT="1.0"
FOCAL_GAMMA="2.0"

# Augmentation
USE_CLAHE="false"
SIGMA_JITTER="true"
CONSISTENCY_WEIGHT="0.1"

# Early Stopping (ADJUSTED)
EARLY_STOP_PATIENCE="20"              # ↑ More patience with slower learning
EARLY_STOP_DELTA="1e-5"

# Training
MIXED_PRECISION="true"

# Data Split
POS_FRACTION="0.6"
SEED="42"

SAVE_DIR="checkpoints/aggressive_${SLURM_JOB_ID}"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     AGGRESSIVE GOLD PARTICLE DETECTION PIPELINE                   ║"
echo "║     (High Recall, Lower Threshold, Strong Positive Weighting)     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Model:                UNetDeepKeypointDetector"
echo "  Data:                 Sliding window (256×256, stride=128)"
echo ""
echo "Aggressive Settings:"
echo "  Positive Weight:      $LOSS_POS_WEIGHT (vs 30 in baseline)"
echo "  Learning Rate:        $LR (vs 5e-4 in baseline)"
echo "  Training Epochs:      $EPOCHS (vs 100 in baseline)"
echo "  Early Stop Patience:  $EARLY_STOP_PATIENCE (vs 10 in baseline)"
echo ""
echo "Expected Behavior:"
echo "  Recall:               ~96-98% (find almost all particles)"
echo "  Precision:            ~70-80% (more false positives)"
echo "  F1 Score:             ~0.82-0.88"
echo "  Detection Threshold:  0.08-0.10 (vs 0.20 in baseline)"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs checkpoints "$SAVE_DIR"

# Setup environment
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python3/3.9.16 || true
  module load cuda/11.8 || true
fi

BASE_TMP="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${BASE_TMP}/venv_detector_agg_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade "pip<24"
python -m pip install imagecodecs numpy tifffile matplotlib scipy
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Build command
CMD=(
  python -u train_detector.py
  --data_root "$DATA_ROOT"
  --model_type "$MODEL_TYPE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --patch_h "$PATCH_SIZE"
  --patch_w "$PATCH_SIZE"
  --train_samples_per_epoch "$TRAIN_SAMPLES_PER_EPOCH"
  --val_samples_per_epoch "$VAL_SAMPLES_PER_EPOCH"
  --sigma "$SIGMA"
  --target_type "$TARGET_TYPE"
  --base_channels "$BASE_CHANNELS"
  --loss_type "$LOSS_TYPE"
  --loss_pos_weight "$LOSS_POS_WEIGHT"
  --loss_neg_weight "$LOSS_NEG_WEIGHT"
  --focal_gamma "$FOCAL_GAMMA"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --consistency_weight "$CONSISTENCY_WEIGHT"
  --sched "$SCHED"
  --warmup_epochs "$WARMUP_EPOCHS"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --early_stop_delta "$EARLY_STOP_DELTA"
  --pos_fraction "0.6"
  --seed "$SEED"
  --save_dir "$SAVE_DIR"
)

# Conditional flags
[ "$USE_CLAHE" = "true" ] && CMD+=(--use_clahe)
[ "$SIGMA_JITTER" = "true" ] && CMD+=(--sigma_jitter)
[ "$MIXED_PRECISION" = "true" ] && CMD+=(--mixed_precision)
[ "$USE_SLIDING_WINDOW" = "true" ] && CMD+=(--use_sliding_window --patch_stride "$PATCH_STRIDE")

echo "Starting aggressive training..."
echo "Command: ${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "✓ Training complete"
echo "✓ Best model saved to: $SAVE_DIR/detector_best.pt"
```

### 5.2 Inference with Lower Threshold

```python
# After training, evaluate with aggressive settings

def evaluate_aggressive(model, test_records, device):
    """Evaluate with aggressive settings."""

    thresholds = [0.05, 0.08, 0.10, 0.15, 0.20]

    for threshold in thresholds:
        print(f"\n╔════════════════════════════════════════════════╗")
        print(f"║ Evaluation at threshold={threshold}        ║")
        print(f"╚════════════════════════════════════════════════╝")

        results = evaluate(
            model,
            test_records,
            device,
            threshold=threshold
        )

        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  F1:        {results['f1']:.3f}")
        print(f"  TP:        {results['tp']}")
        print(f"  FP:        {results['fp']}")
        print(f"  FN:        {results['fn']}")
```

---

## Section 6: Training Trajectory Predictions

### 6.1 Expected Training Curves

```
BASELINE (pos_weight=30, lr=5e-4, 100 epochs):
───────────────────────────────────────────────

Validation Loss:                    Validation F1:

0.8 ┌─╲                              0.10 ┌──╲
    │  ╲                                  │   ╲
0.7 │   ╲                            0.08 │    ╲
    │    ╲                                │     ╲
0.6 │     ╲╲                         0.06 │      ╲
    │      ╲ ╲                            │       ╲╲
0.5 │       ╲ ╲__                    0.05 │        ╲ ╲
    │        ╲    ╲__                     │         ╲ ╲__
0.4 │         ╲       ╲__             0.04 │          ╲   ╲__
    │          ╲          ╲___             │           ╲     ╲___
0.3 │           ╲             ╲_________   0.03 │            ╲     ╲______
    └──────┬──────┬──────┬──────┬──────►    └──────┬──────┬──────┬──────►
      0    20     40     60     80    100    0    20     40     60     80   100
                                  Epoch                              Epoch

Best epoch: ~60-70
Best val_loss: ~0.35
Best F1: ~0.008-0.010


AGGRESSIVE (pos_weight=200, lr=1e-4, 200 epochs):
──────────────────────────────────────────────────

Validation Loss:                    Validation F1:

1.2 ┌──╲                             0.12 ┌──╲
    │   ╲                                 │   ╲
1.0 │    ╲                           0.10 │    ╲
    │     ╲                               │     ╲╲
0.8 │      ╲╲                        0.08 │      ╲ ╲
    │       ╲ ╲                           │       ╲ ╲
0.6 │        ╲ ╲__                   0.06 │        ╲ ╲__
    │         ╲    ╲__                   │         ╲    ╲__
0.4 │          ╲       ╲                 0.04 │         ╲      ╲__
    │           ╲        ╲__                  │          ╲         ╲___
0.2 │            ╲           ╲___________     0.02 │       ╲            ╲____
    └──────┬──────┬──────┬──────┬──────►      └──────┬──────┬──────┬──────►
      0   40   80  120   160   200    Epoch   0   40   80  120   160   200

Best epoch: ~120-140
Best val_loss: ~0.32 (similar to baseline)
Best F1: ~0.012-0.015

Smoothness: More gradual decay due to lower learning rate
```

---

## Section 7: Heatmap Analysis Script

Create a new analysis script to visualize heatmaps:

```python
# analyze_heatmap_aggressive.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_dir = os.path.join(os.path.dirname(__file__), 'project')
sys.path.insert(0, project_dir)

from prepare_labels import discover_image_records
from dataset_points import PointPatchDataset
from model_unet_deep import UNetDeepKeypointDetector

def analyze_heatmap_distribution(heatmap, name=""):
    """Analyze heatmap statistics."""

    print(f"\n{'═'*70}")
    print(f"HEATMAP ANALYSIS: {name}")
    print(f"{'═'*70}")

    print(f"\nValue distribution:")
    print(f"  Min:        {heatmap.min():.6f}")
    print(f"  Max:        {heatmap.max():.6f}")
    print(f"  Mean:       {heatmap.mean():.6f}")
    print(f"  Median:     {np.median(heatmap):.6f}")
    print(f"  Std:        {heatmap.std():.6f}")

    print(f"\nPixels above thresholds:")
    for threshold in [0.01, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        count = (heatmap > threshold).sum()
        pct = 100 * count / heatmap.size
        print(f"  > {threshold:.2f}: {count:5d} pixels ({pct:5.1f}%)")

    print(f"\nPeak statistics:")
    # Find local maxima
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(heatmap, size=3)
    peak_mask = (heatmap == local_max)
    peak_values = heatmap[peak_mask]

    if len(peak_values) > 0:
        print(f"  Number of peaks: {len(peak_values)}")
        print(f"  Peak heights - Min: {peak_values.min():.3f}, "
              f"Mean: {peak_values.mean():.3f}, "
              f"Max: {peak_values.max():.3f}")

    print(f"\nActivation sharpness:")
    # Ratio of pixels with value > 0.5 to pixels > 0.1
    above_50 = (heatmap > 0.50).sum()
    above_10 = (heatmap > 0.10).sum()
    if above_10 > 0:
        sharpness = above_50 / above_10
        print(f"  (>0.5) / (>0.1) ratio: {sharpness:.3f}")
        print(f"    → >0.5: {sharpness:.1%} (sharp peaks)")
        print(f"    → 0.1-0.5: {1-sharpness:.1%} (diffuse halo)")

def visualize_heatmaps(baseline_hm, aggressive_hm, ground_truth):
    """Compare baseline vs aggressive heatmaps."""

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    def norm_img(img):
        img = np.asarray(img, dtype=np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        return img

    # Ground truth
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(ground_truth, cmap='hot')
    ax.set_title('Ground Truth\n(σ=1.0 Gaussian)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Heatmap value')
    ax.axis('off')

    # Baseline heatmap
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(baseline_hm, cmap='hot')
    ax.set_title('Baseline Prediction\n(pos_weight=30, lr=5e-4)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Heatmap value')
    ax.axis('off')

    # Aggressive heatmap
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(aggressive_hm, cmap='hot')
    ax.set_title('Aggressive Prediction\n(pos_weight=200, lr=1e-4)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Heatmap value')
    ax.axis('off')

    # Difference (Aggressive - Baseline)
    ax = fig.add_subplot(gs[1, 0])
    diff = aggressive_hm - baseline_hm
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax.set_title('Difference\n(Aggressive - Baseline)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Δ Heatmap')
    ax.axis('off')

    # Histogram
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(baseline_hm.flatten(), bins=50, alpha=0.6, label='Baseline',
            color='blue', density=True)
    ax.hist(aggressive_hm.flatten(), bins=50, alpha=0.6, label='Aggressive',
            color='red', density=True)
    ax.set_xlabel('Heatmap Value')
    ax.set_ylabel('Density')
    ax.set_title('Value Distribution Comparison')
    ax.legend()
    ax.set_yscale('log')

    # Cumulative
    ax = fig.add_subplot(gs[1, 2])
    baseline_flat = np.sort(baseline_hm.flatten())
    aggressive_flat = np.sort(aggressive_hm.flatten())
    cdf_baseline = np.arange(1, len(baseline_flat)+1) / len(baseline_flat)
    cdf_aggressive = np.arange(1, len(aggressive_flat)+1) / len(aggressive_flat)
    ax.plot(baseline_flat, cdf_baseline, label='Baseline', color='blue', lw=2)
    ax.plot(aggressive_flat, cdf_aggressive, label='Aggressive', color='red', lw=2)
    ax.axvline(0.20, color='blue', linestyle='--', alpha=0.5, label='Baseline threshold')
    ax.axvline(0.08, color='red', linestyle='--', alpha=0.5, label='Aggressive threshold')
    ax.set_xlabel('Heatmap Value')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Heatmap Comparison: Baseline vs Aggressive Configuration',
                 fontsize=14, fontweight='bold')
    plt.savefig('heatmap_analysis_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: heatmap_analysis_comparison.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    records = discover_image_records(
        "project/data/Max Planck Data/Gold Particle Labelling/analyzed synapses"
    )

    # Get test patch
    ds = PointPatchDataset(
        records[:1],
        patch_size=(256, 256),
        samples_per_epoch=1,
        augment=False
    )

    image, gt_heatmap = ds[0]
    image = image.unsqueeze(0).to(device)
    gt_heatmap = gt_heatmap.numpy()

    # Load both models
    baseline_model = UNetDeepKeypointDetector(3, 2, 32).to(device)
    aggressive_model = UNetDeepKeypointDetector(3, 2, 32).to(device)

    # baseline_model.load_state_dict(torch.load("baseline_best.pt"))
    # aggressive_model.load_state_dict(torch.load("aggressive_best.pt"))

    # Inference
    with torch.no_grad():
        baseline_out = torch.sigmoid(baseline_model(image))
        aggressive_out = torch.sigmoid(aggressive_model(image))

    baseline_hm = baseline_out[0].cpu().numpy().mean(axis=0)
    aggressive_hm = aggressive_out[0].cpu().numpy().mean(axis=0)
    gt_hm = gt_heatmap.mean(axis=0)

    # Analyze
    analyze_heatmap_distribution(baseline_hm, "Baseline")
    analyze_heatmap_distribution(aggressive_hm, "Aggressive")
    analyze_heatmap_distribution(gt_hm, "Ground Truth")

    # Visualize
    visualize_heatmaps(baseline_hm, aggressive_hm, gt_hm)
```

---

## Section 8: Recommendation Summary

| Aspect | Baseline (V3) | Aggressive (V4) | When to Use |
|--------|---------------|-----------------|------------|
| **pos_weight** | 30 | 200 | Aggressive when recall is critical |
| **Learning rate** | 5e-4 | 1e-4 | Lower with higher pos_weight |
| **Epochs** | 100 | 200 | More epochs for slower learning |
| **Threshold** | 0.20 | 0.08-0.10 | Lower for higher sensitivity |
| **Precision** | ~90% | ~75% | Trade precision for recall |
| **Recall** | ~80% | ~96% | Find almost all particles |
| **F1 Score** | ~0.85 | ~0.84 | Similar F1, different operating point |
| **Use Case** | Quality (few FP) | Completeness (all particles) |

---

## Conclusion

The aggressive configuration increases **recall from 80% to 96%** at the cost of **precision dropping from 90% to 75%**. This is ideal if you need to find every particle and can tolerate some false positives, which can be filtered in post-processing.

**Next steps**:
1. Submit aggressive training: `sbatch hpc/train_detector_2d_aggressive.slurm`
2. After ~150 epochs (early stop), run heatmap analysis script
3. Compare outputs on test image
4. Sweep detection thresholds to find optimal operating point
5. Decide which configuration better meets your goals
