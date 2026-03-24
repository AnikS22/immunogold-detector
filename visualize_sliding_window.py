"""
Visualize sliding window patch extraction strategy.
Creates a presentation-ready diagram showing the patching strategy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def create_sliding_window_visualization():
    """Create comprehensive sliding window visualization."""

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # =========================================================================
    # Panel 1: Full Image with Patch Grid
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Create synthetic EM image
    np.random.seed(42)
    image = np.random.normal(0.5, 0.1, (2048, 2048))
    image = np.clip(image, 0, 1)

    # Add some "particles" as dark spots
    for _ in range(40):
        y, x = np.random.randint(50, 1998), np.random.randint(50, 1998)
        yy, xx = np.ogrid[-15:16, -15:16]
        mask = xx*xx + yy*yy <= 225
        y_slice = image[y-15:y+16, x-15:x+16]
        y_slice[mask] = 0.2

    ax1.imshow(image, cmap='gray', extent=[0, 2048, 2048, 0])
    ax1.set_title('Full EM Image (2048×2048) with Sliding Window Patch Grid\n'
                  'Patch Size: 256×256 | Stride: 128 pixels (50% Overlap)',
                  fontsize=14, fontweight='bold', pad=20)

    # Draw patch grid
    patch_size = 256
    stride = 128

    # Vertical lines
    for x in range(0, 2048 - patch_size + 1, stride):
        ax1.axvline(x, color='cyan', linewidth=1, alpha=0.6, linestyle='--')
        ax1.axvline(x + patch_size, color='cyan', linewidth=1, alpha=0.6, linestyle='--')

    # Horizontal lines
    for y in range(0, 2048 - patch_size + 1, stride):
        ax1.axhline(y, color='cyan', linewidth=1, alpha=0.6, linestyle='--')
        ax1.axhline(y + patch_size, color='cyan', linewidth=1, alpha=0.6, linestyle='--')

    # Highlight first patch (red)
    rect1 = patches.Rectangle((0, 0), 256, 256, linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect1)
    ax1.text(128, -80, 'Patch 1\n(0:256, 0:256)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7), color='white', fontweight='bold')

    # Highlight second patch (green, overlapping with first)
    rect2 = patches.Rectangle((128, 0), 256, 256, linewidth=3, edgecolor='lime', facecolor='none', linestyle=':')
    ax1.add_patch(rect2)
    ax1.text(256, -150, 'Patch 2\n(0:256, 128:384)\n50% overlap', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7), color='black', fontweight='bold')

    # Highlight corner patch
    x_max = ((2048 - 256) // 128) * 128
    y_max = ((2048 - 256) // 128) * 128
    rect_last = patches.Rectangle((x_max, y_max), 256, 256, linewidth=3, edgecolor='yellow', facecolor='none')
    ax1.add_patch(rect_last)

    ax1.set_xlim(-200, 2300)
    ax1.set_ylim(2300, -200)
    ax1.set_xlabel('X (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (pixels)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # Add text box with statistics
    stats_text = (
        'PATCH GRID STATISTICS:\n'
        '━━━━━━━━━━━━━━━━━━━━━\n'
        f'Image size: 2048 × 2048\n'
        f'Patch size: 256 × 256\n'
        f'Stride: 128 pixels\n'
        f'Overlap: 50% (128/256)\n'
        f'\n'
        f'Patches per dimension: (2048-256)/128 + 1 = 15\n'
        f'Total patches per image: 15 × 15 = 225\n'
        f'Usable patches (no edge): ~200\n'
        f'\n'
        f'Per epoch: 200 patches available\n'
        f'Training: 2,048 samples/epoch (with replacement)\n'
        f'Data amplification: 2,048/200 = 10.24×'
    )
    ax1.text(2100, 100, stats_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')

    # =========================================================================
    # Panel 2: Zoom on Overlap Region
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Show zoom of overlap region (patches 1 and 2 overlap)
    zoom_region = image[0:384, 0:384]
    ax2.imshow(zoom_region, cmap='gray', extent=[0, 384, 384, 0])

    # Draw patch boundaries
    ax2.axvline(256, color='red', linewidth=3, label='Patch 1 (0:256, 0:256)', linestyle='-')
    ax2.axhline(256, color='red', linewidth=3, linestyle='-')

    ax2.axvline(128, color='lime', linewidth=3, label='Patch 2 (0:256, 128:384)', linestyle=':')
    ax2.axhline(128, color='lime', linewidth=3, linestyle=':')
    ax2.axvline(384, color='lime', linewidth=3, linestyle=':')

    # Highlight overlap region
    overlap = patches.Rectangle((128, 0), 128, 256, linewidth=2, edgecolor='yellow',
                               facecolor='yellow', alpha=0.3)
    ax2.add_patch(overlap)
    ax2.text(192, 128, 'OVERLAP\nREGION\n(128 pixels)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax2.set_title('Zoom: Patch Overlap Region\n'
                  'Both patches see same particles (128-pixel width overlap)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y (pixels)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Add annotations
    ax2.annotate('', xy=(128, -30), xytext=(0, -30),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(64, -60, '128 pixels', ha='center', fontsize=10, color='red', fontweight='bold')

    ax2.annotate('', xy=(256, -30), xytext=(128, -30),
                arrowprops=dict(arrowstyle='<->', color='lime', lw=2))
    ax2.text(192, -60, '128 pixels\n(overlap)', ha='center', fontsize=10, color='lime', fontweight='bold')

    # =========================================================================
    # Panel 3: Training Strategy
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    strategy_text = """
SLIDING WINDOW PATCHING STRATEGY
═══════════════════════════════════════════════════════════════════

WHY SLIDING WINDOW?
───────────────────
✓ More patches available: 200 per image (vs 10 random)
✓ Better coverage: Overlapping ensures particles not cut at boundaries
✓ Richer gradients: Each particle seen from multiple angles
✓ Data amplification: 2,048 samples/epoch from only ~200 unique patches

HOW IT WORKS (Per Epoch):
────────────────────────
1. Precompute all patch locations:
   - 7 training images × 200 patches/image = 1,400 locations available

2. For 2,048 training samples:
   - Randomly sample from 1,400 locations (with replacement)
   - Each sample can be selected multiple times per epoch
   - Each gets unique augmentation (elastic, blur, gamma, etc.)

3. Result: 2,048 × 4.3 augmentations = 8,806 unique augmented views

DATA AMPLIFICATION EXAMPLE:
──────────────────────────
Baseline (random sampling):
  └─ 10 patches/epoch × 100 epochs = 1,000 total patches

Sliding window (our approach):
  └─ 2,048 patches/epoch × 100 epochs × 4.3 aug/patch = 880,640 views
  └─ Data amplification: 880× compared to random sampling!

OVERLAP BENEFIT:
────────────────
When stride < patch_size (128 < 256):
  ✓ Each particle appears in 1-4 different patches
  ✓ Model sees particle from different positions
  ✓ Richer training signal for gradient computation
  ✓ Better edge detection (particles not cut off)
"""

    ax3.text(0.05, 0.95, strategy_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('Sliding Window Patch Extraction for EM Image Training',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig('sliding_window_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sliding_window_visualization.png (presentation-ready)")

    return fig

def create_comparison_visualization():
    """Compare random sampling vs sliding window."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ========== Left: Random Sampling ==========
    ax = axes[0]
    np.random.seed(42)
    image = np.random.normal(0.5, 0.1, (2048, 2048))
    image = np.clip(image, 0, 1)

    # Add particles
    for _ in range(40):
        y, x = np.random.randint(50, 1998), np.random.randint(50, 1998)
        yy, xx = np.ogrid[-15:16, -15:16]
        mask = xx*xx + yy*yy <= 225
        y_slice = image[y-15:y+16, x-15:x+16]
        y_slice[mask] = 0.2

    ax.imshow(image, cmap='gray')
    ax.set_title('RANDOM SAMPLING (Baseline)\n'
                 'Patch Size: 512×512 | Random Selection',
                 fontsize=12, fontweight='bold')

    # Draw 10 random patches
    np.random.seed(123)
    for i in range(10):
        y0 = np.random.randint(0, 2048 - 512)
        x0 = np.random.randint(0, 2048 - 512)
        rect = patches.Rectangle((x0, y0), 512, 512, linewidth=2,
                                edgecolor=plt.cm.tab10(i), facecolor='none')
        ax.add_patch(rect)

    stats = (
        'STATISTICS:\n'
        '─────────────\n'
        'Patches per epoch: 10\n'
        'Patch size: 512×512\n'
        'Coverage: <1%\n'
        'Overlap: None\n'
        'Data amplification: 1×\n'
        'Expected epochs: 100\n'
        'Total views: 1,000'
    )
    ax.text(2100, 200, stats, fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
           verticalalignment='top')
    ax.set_xlim(-200, 2400)
    ax.set_ylim(2400, -200)

    # ========== Right: Sliding Window ==========
    ax = axes[1]
    ax.imshow(image, cmap='gray')
    ax.set_title('SLIDING WINDOW (Our Approach)\n'
                 'Patch Size: 256×256 | Stride: 128 (50% Overlap)',
                 fontsize=12, fontweight='bold')

    # Draw grid
    patch_size = 256
    stride = 128
    colors = plt.cm.get_cmap('rainbow', 15)

    for i, x in enumerate(range(0, 2048 - patch_size + 1, stride)):
        for j, y in enumerate(range(0, 2048 - patch_size + 1, stride)):
            color = colors((i + j) % 15)
            rect = patches.Rectangle((x, y), patch_size, patch_size, linewidth=1,
                                    edgecolor=color, facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    stats = (
        'STATISTICS:\n'
        '─────────────\n'
        'Patches per epoch: ~200\n'
        'Patch size: 256×256\n'
        'Coverage: ~100%\n'
        'Overlap: 50% (128px)\n'
        'Data amplification: 880×*\n'
        'Expected epochs: 100\n'
        'Total views: 880,640*\n'
        '\n*With 4.3 aug/patch'
    )
    ax.text(2100, 200, stats, fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
           verticalalignment='top')
    ax.set_xlim(-200, 2400)
    ax.set_ylim(2400, -200)

    fig.suptitle('Patch Extraction Strategy Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('patch_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: patch_strategy_comparison.png (comparison diagram)")

    return fig

if __name__ == "__main__":
    print("Creating sliding window visualization for presentation...\n")

    fig1 = create_sliding_window_visualization()
    fig2 = create_comparison_visualization()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. sliding_window_visualization.png - Detailed explanation")
    print("  2. patch_strategy_comparison.png - Side-by-side comparison")
    print("\nBoth are 300 DPI (presentation-quality)")
