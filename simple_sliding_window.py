"""
Simple, clean sliding window diagram for presentations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ============================================================================
# LEFT: Simple diagram showing the concept
# ============================================================================

ax1.set_xlim(-100, 2150)
ax1.set_ylim(2150, -100)
ax1.set_aspect('equal')

# Draw image boundary
img_rect = patches.Rectangle((0, 0), 2048, 2048, linewidth=4,
                             edgecolor='black', facecolor='lightgray', alpha=0.3)
ax1.add_patch(img_rect)
ax1.text(1024, -80, 'Full EM Image: 2048 × 2048 pixels',
        ha='center', fontsize=14, fontweight='bold')

# Draw only 3 patches clearly to avoid clutter
patch_positions = [
    (0, 0, 'Patch 1'),
    (256, 0, 'Patch 2'),
    (512, 0, 'Patch 3'),
    (0, 256, 'Patch 4'),
    (256, 256, 'Patch 5'),
]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for idx, (x, y, label) in enumerate(patch_positions):
    rect = patches.Rectangle((x, y), 256, 256, linewidth=2.5,
                             edgecolor=colors[idx], facecolor=colors[idx], alpha=0.4)
    ax1.add_patch(rect)
    ax1.text(x + 128, y + 128, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color='black')

# Draw overlap arrows
ax1.annotate('', xy=(256, -40), xytext=(0, -40),
            arrowprops=dict(arrowstyle='<->', color='red', lw=3))
ax1.text(128, -65, '128 px\nstride', ha='center', fontsize=11,
        fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Overlap highlight
overlap = patches.Rectangle((128, 0), 128, 256, linewidth=3,
                           edgecolor='orange', facecolor='orange', alpha=0.2, linestyle='--')
ax1.add_patch(overlap)
ax1.text(192, 380, '50% Overlap', ha='center', fontsize=12,
        fontweight='bold', color='orange',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax1.set_title('Sliding Window Patching\n(Stride = 128 pixels, 50% overlap)',
             fontsize=13, fontweight='bold', pad=20)
ax1.axis('off')

# ============================================================================
# RIGHT: Key numbers comparison
# ============================================================================

ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Title
ax2.text(5, 9.5, 'Why Sliding Window Works', ha='center', fontsize=16, fontweight='bold')

# Left column: Random Sampling
y_pos = 8.5
ax2.text(2.5, y_pos, 'Random Sampling', ha='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFB6B6', alpha=0.9))
y_pos -= 0.8

stats_random = [
    '✗ Patches/epoch: 10',
    '✗ Coverage: <1%',
    '✗ Data: 1,000 views',
    '✗ Particles: Often cut at edges',
]

for stat in stats_random:
    ax2.text(2.5, y_pos, stat, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    y_pos -= 0.6

# Right column: Sliding Window
y_pos = 8.5
ax2.text(7.5, y_pos, 'Sliding Window (Ours)', ha='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#B6E5B6', alpha=0.9))
y_pos -= 0.8

stats_sliding = [
    '✓ Patches/epoch: 200',
    '✓ Coverage: 100%',
    '✓ Data: 880,640 views*',
    '✓ All particles in multiple patches',
]

for stat in stats_sliding:
    ax2.text(7.5, y_pos, stat, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    y_pos -= 0.6

# Middle arrow
ax2.annotate('', xy=(6.5, 5), xytext=(3.5, 5),
            arrowprops=dict(arrowstyle='->', color='green', lw=4))

# Key benefit
y_pos = 2.5
ax2.text(5, y_pos, '20× More Training Data', ha='center', fontsize=14, fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.9, pad=1))

# Footnote
ax2.text(5, 0.8, '*With 4.3 augmentations per patch × 100 epochs',
        ha='center', fontsize=9, style='italic', color='gray')

plt.suptitle('Sliding Window: How We Get More Training Data from Limited EM Images',
            fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('simple_sliding_window.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: simple_sliding_window.png")
print("\nDimensions: 16×7 inches, 300 DPI")
print("Perfect for presentation slides!")
