"""
Simple sliding window overlay on actual image.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create a simple synthetic EM image
np.random.seed(42)
image = np.random.normal(0.5, 0.1, (800, 800))
image = np.clip(image, 0, 1)

# Add some dark spots (simulated particles)
for _ in range(20):
    y, x = np.random.randint(50, 750), np.random.randint(50, 750)
    yy, xx = np.ogrid[-15:16, -15:16]
    mask = xx*xx + yy*yy <= 200
    image[y-15:y+16, x-15:x+16][mask] = 0.2

# ============================================================================
# LEFT: With grid overlay
# ============================================================================
ax = axes[0]
ax.imshow(image, cmap='gray')
ax.set_title('Sliding Window Overlay\n(256×256 patches, stride 128)', fontsize=12, fontweight='bold')
ax.set_xlabel('Image width')
ax.set_ylabel('Image height')

# Draw grid - showing patches
patch_size = 256
stride = 128

# Scale to 800x800 display (from 2048x2048 actual)
scale = 800 / 2048

for x in range(0, 800 - int(patch_size*scale) + 1, int(stride*scale)):
    for y in range(0, 800 - int(patch_size*scale) + 1, int(stride*scale)):
        rect = patches.Rectangle((x, y), int(patch_size*scale), int(patch_size*scale),
                                linewidth=1.5, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)

# Highlight overlap
overlap_width = int(128 * scale)
for x in range(int(stride*scale), 800 - int(patch_size*scale), int(stride*scale)):
    rect = patches.Rectangle((x-overlap_width/2, 0), overlap_width, 800,
                            linewidth=0, edgecolor='none', facecolor='yellow', alpha=0.1)
    ax.add_patch(rect)

ax.text(400, -60, '← Each patch is 256×256, shifted by 128 pixels\n← Yellow bands = overlapping regions',
       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ============================================================================
# RIGHT: Show the concept
# ============================================================================
ax = axes[1]
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Title
ax.text(5, 9.3, 'How It Works', fontsize=13, fontweight='bold', ha='center')

# Concept boxes
y = 8.5
ax.add_patch(patches.FancyBboxPatch((0.5, y-0.4), 9, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='#FFE5E5', linewidth=2))
ax.text(5, y, '1 Image → Many Overlapping Patches', ha='center', va='center', fontsize=11, fontweight='bold')

y = 7.5
ax.text(5, y, '10 random patches', ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='#FFB6B6', alpha=0.9))
ax.text(8, y, '❌ Only 1% coverage', ha='left', fontsize=9, style='italic')

y = 6.5
ax.text(5, y, '200 sliding patches', ha='center', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='#B6E5B6', alpha=0.9))
ax.text(8, y, '✓ Full coverage', ha='left', fontsize=9, style='italic')

# Arrow
ax.annotate('', xy=(5, 5.8), xytext=(5, 6.2),
           arrowprops=dict(arrowstyle='->', lw=3, color='green'))

y = 5.2
ax.add_patch(patches.FancyBboxPatch((0.5, y-0.6), 9, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor='green', facecolor='#E8F5E9', linewidth=2))
ax.text(5, y+0.1, 'Each particle seen in ~4 patches', ha='center', fontsize=11, fontweight='bold', color='darkgreen')
ax.text(5, y-0.35, '(from different angles = better training)', ha='center', fontsize=9, style='italic', color='darkgreen')

y = 3.5
ax.text(5, y, '880,640 training views', ha='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.7))
ax.text(5, y-0.6, '(200 patches × 100 epochs × 4.3 augmentations)',
       ha='center', fontsize=8, style='italic', color='gray')

y = 1.5
ax.add_patch(patches.FancyBboxPatch((0.5, y-0.4), 9, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='darkgreen', facecolor='lightgreen', linewidth=2))
ax.text(5, y, '20× More Training Data', ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('simple_overlay.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: simple_overlay.png")
print("\nFeatures:")
print("  ✓ Actual image with overlay grid")
print("  ✓ Yellow bands show overlap")
print("  ✓ Simple explanation on right")
print("  ✓ Not complicated - just the essentials")
