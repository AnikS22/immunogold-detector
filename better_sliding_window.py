"""
Better, clearer sliding window diagram - not overcomplicated but complete.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(16, 10))

# Main title
fig.suptitle('Sliding Window Patching: Extract More Data from Limited Images',
            fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# TOP: Visual explanation with actual example
# ============================================================================
ax_top = plt.subplot(2, 1, 1)
ax_top.set_xlim(-300, 1200)
ax_top.set_ylim(-200, 850)
ax_top.set_aspect('equal')
ax_top.axis('off')

# Draw simplified image (not full 2048 to keep clean)
img_w, img_h = 800, 800
img_rect = patches.Rectangle((0, 0), img_w, img_h, linewidth=3,
                             edgecolor='black', facecolor='#E8E8E8', alpha=0.3)
ax_top.add_patch(img_rect)

# Label image
ax_top.text(img_w/2, -50, 'Single EM Image\n(2048 × 2048 pixels)',
           ha='center', fontsize=12, fontweight='bold')

# Draw 4 patches in 2x2 grid showing overlap
patch_size = 256
stride = 128

# Scale for visualization
scale = img_w / 2048

patch_data = [
    (0, 0, '#FF6B6B', 'Patch 1'),
    (stride, 0, '#4ECDC4', 'Patch 2'),
    (0, stride, '#45B7D1', 'Patch 3'),
    (stride, stride, '#FFA07A', 'Patch 4'),
]

for x, y, color, label in patch_data:
    x_scaled = x * scale
    y_scaled = y * scale
    size_scaled = patch_size * scale

    rect = patches.Rectangle((x_scaled, y_scaled), size_scaled, size_scaled,
                             linewidth=2.5, edgecolor=color, facecolor=color, alpha=0.25)
    ax_top.add_patch(rect)

    # Label in center of patch
    ax_top.text(x_scaled + size_scaled/2, y_scaled + size_scaled/2, label,
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=color)

# Draw stride arrow
stride_scaled = stride * scale
ax_top.annotate('', xy=(stride_scaled, -100), xytext=(0, -100),
               arrowprops=dict(arrowstyle='<->', color='red', lw=3))
ax_top.text(stride_scaled/2, -140, 'Stride = 128 px\n(50% overlap)',
           ha='center', fontsize=11, fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, pad=0.5))

# Draw patch size annotation
ax_top.annotate('', xy=(-50, size_scaled * scale), xytext=(-50, 0),
               arrowprops=dict(arrowstyle='<->', color='blue', lw=2.5))
ax_top.text(-100, size_scaled * scale / 2, '256 px',
           ha='center', va='center', fontsize=10, fontweight='bold', color='blue',
           rotation=90,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Highlight overlap region
overlap_scale = (stride * scale)
overlap = patches.Rectangle((overlap_scale, 0), overlap_scale, overlap_scale,
                           linewidth=2, edgecolor='orange', facecolor='orange',
                           alpha=0.15, linestyle='--')
ax_top.add_patch(overlap)
ax_top.text(overlap_scale + overlap_scale/2, overlap_scale/2,
           'OVERLAPPING\nREGION',
           ha='center', va='center', fontsize=10, fontweight='bold',
           color='darkorange',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ============================================================================
# BOTTOM: How it translates to more data
# ============================================================================
ax_bot = plt.subplot(2, 1, 2)
ax_bot.set_xlim(0, 10)
ax_bot.set_ylim(0, 6)
ax_bot.axis('off')

# Title for bottom section
ax_bot.text(5, 5.5, 'Why Sliding Window = More Training Data',
           ha='center', fontsize=14, fontweight='bold')

# Left column
ax_bot.text(1.5, 4.8, 'WITHOUT Sliding Window', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFB6B6', alpha=0.8, pad=0.7))

without_text = [
    '• 10 random patches per epoch',
    '• Patches never overlap',
    '• Particles cut at boundaries',
    '• Coverage: <1% of image',
    '',
    'Total data = 10 patches × 100 epochs',
    '= 1,000 patches'
]

y = 4.3
for text in without_text:
    if text == '':
        y -= 0.2
    else:
        ax_bot.text(1.5, y, text, ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        y -= 0.3

# Arrow between
arrow = FancyArrowPatch((3, 2.5), (7, 2.5), arrowstyle='->', mutation_scale=40,
                       linewidth=4, color='green', alpha=0.7)
ax_bot.add_patch(arrow)
ax_bot.text(5, 2.8, '20× MORE DATA', ha='center', fontsize=13, fontweight='bold',
           color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=0.8))

# Right column
ax_bot.text(8.5, 4.8, 'WITH Sliding Window', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#B6E5B6', alpha=0.8, pad=0.7))

with_text = [
    '• 200 overlapping patches per epoch',
    '• Patches overlap 50% (stride=128)',
    '• Particles in multiple patches',
    '• Coverage: 100% of image',
    '',
    'Total data = 200 patches × 100 epochs',
    '× 4.3 augmentations = 880,640 views'
]

y = 4.3
for text in with_text:
    if text == '':
        y -= 0.2
    else:
        ax_bot.text(8.5, y, text, ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        y -= 0.3

# Bottom key benefit
ax_bot.text(5, 0.6, '✓ Better training: Same particles seen from different angles',
           ha='center', fontsize=11, fontweight='bold', style='italic',
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9, pad=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('better_sliding_window.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: better_sliding_window.png")
print("\nImprovements:")
print("  ✓ Clear visual of actual patches with overlap")
print("  ✓ Shows stride and patch size")
print("  ✓ Highlights overlap region")
print("  ✓ Complete comparison: with vs without")
print("  ✓ Explains the benefit")
print("  ✓ Still simple, not overcomplicated")
