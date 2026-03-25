"""Rich augmentation library for EM immunogold particle detection.

Uses only numpy and scipy (no external deps for HPC compatibility).
"""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage


class ElasticDeform:
    """Elastic deformation of image and heatmap using displacement fields."""

    def __init__(self, alpha: float = 30.0, sigma: float = 5.0) -> None:
        self.alpha = float(alpha)
        self.sigma = float(sigma)

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation to both image and heatmap.

        Args:
            image: shape (C, H, W)
            heatmap: shape (C, H, W)
            rng: random generator

        Returns:
            deformed_image, deformed_heatmap
        """
        c, h, w = image.shape
        dy = rng.normal(0, self.sigma, size=(h, w))
        dx = rng.normal(0, self.sigma, size=(h, w))

        # Smooth displacement fields
        dy = ndimage.gaussian_filter(dy, sigma=self.sigma)
        dx = ndimage.gaussian_filter(dx, sigma=self.sigma)

        # Scale
        dy = dy * self.alpha / (self.sigma * 3)
        dx = dx * self.alpha / (self.sigma * 3)

        # Create coordinate maps
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        y_deform = (y + dy).astype(np.float32)
        x_deform = (x + dx).astype(np.float32)

        # Apply to image
        out_image = np.zeros_like(image, dtype=np.float32)
        for i in range(c):
            out_image[i] = ndimage.map_coordinates(
                image[i], [y_deform, x_deform], order=1, mode="constant", cval=0.0
            )

        # Apply to heatmap
        out_heatmap = np.zeros_like(heatmap, dtype=np.float32)
        for i in range(heatmap.shape[0]):
            out_heatmap[i] = ndimage.map_coordinates(
                heatmap[i], [y_deform, x_deform], order=1, mode="constant", cval=0.0
            )

        return out_image, out_heatmap


class GaussianNoise:
    """Additive Gaussian noise to image only."""

    def __init__(self, sigma_range: Tuple[float, float] = (0.01, 0.04)) -> None:
        self.sigma_min, self.sigma_max = sigma_range

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        sigma = float(rng.uniform(self.sigma_min, self.sigma_max))
        noise = rng.normal(0, sigma, size=image.shape)
        image_out = np.clip(image + noise, 0.0, 1.0)
        return image_out, heatmap


class CLAHEPreprocess:
    """Contrast-Limited Adaptive Histogram Equalization.

    Can be applied at image load time or during augmentation.
    Uses sliding window approach (pure numpy, no skimage).
    """

    def __init__(self, tile_size: int = 64, clip_limit: float = 2.0) -> None:
        self.tile_size = int(tile_size)
        self.clip_limit = float(clip_limit)

    def _clahe_tile(self, tile: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a single tile."""
        tile = np.asarray(tile, dtype=np.float32)
        if tile.size == 0 or tile.max() == tile.min():
            return tile

        # Normalize to 0-255 range
        tile_min, tile_max = tile.min(), tile.max()
        tile_norm = ((tile - tile_min) / (tile_max - tile_min) * 255).astype(np.uint8)

        # Simple histogram equalization with clipping
        hist, _ = np.histogram(tile_norm, bins=256, range=(0, 256))
        cdf = np.cumsum(hist)
        cdf_min = cdf[cdf > 0].min()
        cdf = (cdf - cdf_min) * 255 // (cdf[-1] - cdf_min + 1e-10)

        # Apply clip limit
        cdf = np.minimum(cdf, self.clip_limit * 255 / len(hist))
        cdf = (cdf - cdf.min()) * 255 // (cdf.max() - cdf.min() + 1e-10)

        tile_eq = cdf[tile_norm]
        tile_eq = tile_eq.astype(np.float32) / 255.0
        return tile_eq * (tile_max - tile_min) + tile_min

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply CLAHE to image only."""
        c, h, w = image.shape
        image_out = np.zeros_like(image, dtype=np.float32)

        for ch in range(c):
            ch_img = image[ch]
            # Apply CLAHE in tiles
            for i in range(0, h, self.tile_size):
                for j in range(0, w, self.tile_size):
                    i_end = min(i + self.tile_size, h)
                    j_end = min(j + self.tile_size, w)
                    tile = ch_img[i:i_end, j:j_end]
                    image_out[ch, i:i_end, j:j_end] = self._clahe_tile(tile)

        return image_out, heatmap


class GammaCorrection:
    """Random gamma correction to image."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.75, 1.35)) -> None:
        self.gamma_min, self.gamma_max = gamma_range

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        gamma = float(rng.uniform(self.gamma_min, self.gamma_max))
        image_out = np.power(image, gamma)
        return image_out, heatmap


class SaltPepperNoise:
    """Sparse impulse noise (salt and pepper)."""

    def __init__(self, fraction: float = 0.001) -> None:
        self.fraction = float(fraction)

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_out = image.copy()
        mask = rng.random(image.shape) < self.fraction
        image_out[mask] = rng.choice([0.0, 1.0], size=np.sum(mask))
        return image_out, heatmap


class RandomErasing:
    """Zero out random rectangles (occlusion/beam damage)."""

    def __init__(self, max_rectangles: int = 2, max_area_frac: float = 0.1) -> None:
        self.max_rectangles = int(max_rectangles)
        self.max_area_frac = float(max_area_frac)

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        c, h, w = image.shape
        image_out = image.copy()
        heatmap_out = heatmap.copy()

        total_area = h * w
        max_area = int(total_area * self.max_area_frac)
        n_rects = int(rng.integers(0, self.max_rectangles + 1))

        for _ in range(n_rects):
            # Random rectangle dimensions
            rect_h = int(rng.integers(10, min(h // 4 + 1, int(np.sqrt(max_area)) + 1)))
            rect_w = int(rng.integers(10, min(w // 4 + 1, int(np.sqrt(max_area)) + 1)))

            # Random position
            y0 = int(rng.integers(0, h - rect_h + 1))
            x0 = int(rng.integers(0, w - rect_w + 1))
            y1 = y0 + rect_h
            x1 = x0 + rect_w

            # Zero out
            image_out[:, y0:y1, x0:x1] = 0.0
            heatmap_out[:, y0:y1, x0:x1] = 0.0

        return image_out, heatmap_out


class MantisLocalContrast:
    """Mantis-style local contrast enhancement for EM images.

    Enhances local contrast by subtracting a large-kernel Gaussian blur
    (local mean) and normalizing by local standard deviation. This makes
    immunogold particles pop against the background regardless of regional
    intensity variations.
    """

    def __init__(self, kernel_sigma: float = 15.0, strength: float = 0.5) -> None:
        self.kernel_sigma = float(kernel_sigma)
        self.strength = float(strength)

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_out = np.zeros_like(image, dtype=np.float32)
        for ch in range(image.shape[0]):
            channel = image[ch]
            local_mean = ndimage.gaussian_filter(channel, sigma=self.kernel_sigma)
            local_sq_mean = ndimage.gaussian_filter(channel ** 2, sigma=self.kernel_sigma)
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 1e-8))
            enhanced = (channel - local_mean) / (local_std + 1e-8)
            # Normalize enhanced back to [0, 1]
            emin, emax = enhanced.min(), enhanced.max()
            if emax > emin:
                enhanced = (enhanced - emin) / (emax - emin)
            else:
                enhanced = np.zeros_like(enhanced)
            # Blend original and enhanced
            image_out[ch] = np.clip(
                (1.0 - self.strength) * channel + self.strength * enhanced, 0.0, 1.0
            )
        return image_out, heatmap


class GaussianBlur:
    """Gaussian blur for focus/defocus variation in EM.

    NOTE: sigma_range is kept very low (0.3-0.8) to avoid destroying
    small ~1px immunogold particles. Higher values erase the detection signal.
    """

    def __init__(self, sigma_range: Tuple[float, float] = (0.3, 0.8)) -> None:
        self.sigma_min, self.sigma_max = sigma_range

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        sigma = float(rng.uniform(self.sigma_min, self.sigma_max))
        image_out = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            image_out[i] = ndimage.gaussian_filter(image[i], sigma=sigma)
        return image_out, heatmap


class NormalizeRobust:
    """Robust per-patch normalization to prevent saturation/flat patches.

    Uses percentile scaling to restore dynamic range after aggressive transforms
    (e.g. CLAHE/Mantis + brightness/contrast + gamma). Keeps output in [0,1].
    """

    def __init__(self, p_lo: float = 1.0, p_hi: float = 99.0) -> None:
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)

    def __call__(self, image: np.ndarray, heatmap: np.ndarray, rng: Optional[np.random.Generator] = None):
        img = np.asarray(image, dtype=np.float32)
        out = np.zeros_like(img, dtype=np.float32)
        for ch in range(img.shape[0]):
            x = img[ch]
            lo = float(np.quantile(x, self.p_lo / 100.0))
            hi = float(np.quantile(x, self.p_hi / 100.0))
            if hi > lo + 1e-6:
                y = (x - lo) / (hi - lo)
            else:
                # Degenerate case (nearly constant patch). Don't destroy the patch by
                # collapsing it to zeros; keep original values.
                y = x
            y = np.clip(y, 0.0, 1.0)
            # If scaling still saturates most pixels, gently pull away from 1.0.
            if float(np.mean(y >= 0.995)) > 0.85:
                y = np.clip(y * 0.97, 0.0, 1.0)
            out[ch] = y
        return out, heatmap


class Cutout:
    """Single square cutout (dust particle/small defect)."""

    def __init__(self, size_frac: float = 1.0 / 20.0, max_count: int = 1) -> None:
        self.size_frac = float(size_frac)
        self.max_count = int(max_count)

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        c, h, w = image.shape
        image_out = image.copy()
        heatmap_out = heatmap.copy()

        # Small dust particles (fewer, smaller)
        n_cutouts = int(rng.integers(0, self.max_count + 1))
        for _ in range(n_cutouts):
            side = int(h * self.size_frac)
            y0 = int(rng.integers(0, max(1, h - side)))
            x0 = int(rng.integers(0, max(1, w - side)))
            y1 = min(y0 + side, h)
            x1 = min(x0 + side, w)
            image_out[:, y0:y1, x0:x1] = 0.0
            heatmap_out[:, y0:y1, x0:x1] = 0.0

        return image_out, heatmap_out


class MultiScaleSigmaJitter:
    """Jitter sigma at heatmap generation time.

    This is applied at dataset level, not here.
    Used to teach model robustness to particle sigma variation.
    """

    def __init__(self, sigma_range: Tuple[float, float] = (1.5, 3.5)) -> None:
        self.sigma_min, self.sigma_max = sigma_range

    def sample_sigma(self, rng: np.random.Generator) -> float:
        """Return a random sigma value."""
        return float(rng.uniform(self.sigma_min, self.sigma_max))


class BrightnessContrast:
    """Brightness and contrast adjustment (existing style)."""

    def __init__(self, brightness_range: Tuple[float, float] = (-0.08, 0.08),
                 contrast_range: Tuple[float, float] = (0.85, 1.15)) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(
        self, image: np.ndarray, heatmap: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        # If upstream preprocessing (e.g., CLAHE/Mantis) already pushed the patch mean very high,
        # a positive brightness/contrast draw can saturate most pixels to 1.0 (appearing "blank white").
        # We adaptively cap brightness/contrast based on current mean to preserve dynamic range.
        mn_b, mx_b = float(self.brightness_range[0]), float(self.brightness_range[1])
        mn_c, mx_c = float(self.contrast_range[0]), float(self.contrast_range[1])
        m = float(np.asarray(image, dtype=np.float32).mean())
        if m > 0.9:
            # Prevent raising mean further; allow only negative or tiny positive brightness.
            mx_b = min(mx_b, max(0.0, 0.98 - m))
            # Avoid contrast >1 when already near-white.
            mx_c = min(mx_c, 1.0)
        contrast = float(rng.uniform(mn_c, mx_c))
        brightness = float(rng.uniform(mn_b, mx_b))
        image_out = np.clip(image * contrast + brightness, 0.0, 1.0).astype(np.float32)
        # Final guard: avoid near-white drift that can look blank in downstream visualization/training.
        out_mean = float(np.asarray(image_out, dtype=np.float32).mean())
        if out_mean > 0.95:
            shift = out_mean - 0.93
            image_out = np.clip(image_out - shift, 0.0, 1.0).astype(np.float32)
        return image_out, heatmap


def apply_augmentation(
    image: np.ndarray,
    heatmap: np.ndarray,
    rng: np.random.Generator,
    elastic_p: float = 0.3,
    gamma_p: float = 0.5,
    noise_p: float = 0.5,
    salt_pepper_p: float = 0.3,
    cutout_p: float = 0.15,
    blur_p: float = 0.15,
    brightness_contrast_p: float = 0.6,
    flip_p: float = 0.5,
    rot90_p: float = 0.5,
    mantis_p: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply EM-realistic augmentation pipeline.

    Key changes from original:
    - Flips/rot90 raised to 0.5: EM sections have NO canonical orientation,
      these are the most physically justified augmentations
    - Gaussian blur reduced to 0.15 with sigma 0.3-0.8: prevents erasing ~1px particles
    - Elastic deform reduced to 0.3: strong deform can misalign tiny particles
    - Mantis local contrast added: enhances particle visibility
    """
    applied = []

    # GEOMETRIC AUGMENTATIONS FIRST (most impactful, physically justified)

    # Flips — EM has no canonical orientation, these are FREE data diversity
    if rng.random() < flip_p:
        applied.append("flip_x")
        image = image[:, :, ::-1].copy()
        heatmap = heatmap[:, :, ::-1].copy()

    if rng.random() < flip_p:
        applied.append("flip_y")
        image = image[:, ::-1, :].copy()
        heatmap = heatmap[:, ::-1, :].copy()

    # 90-degree rotations — same justification as flips
    if rng.random() < rot90_p:
        k = int(rng.integers(1, 4))
        applied.append(f"rot90_k{k}")
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        heatmap = np.rot90(heatmap, k=k, axes=(1, 2)).copy()

    # REALISTIC EM AUGMENTATIONS

    # Elastic deformation (specimen drift, charging effects)
    if rng.random() < elastic_p:
        applied.append("elastic(alpha=20,sigma=4)")
        elastic = ElasticDeform(alpha=20.0, sigma=4.0)
        image, heatmap = elastic(image, heatmap, rng)

    # Mantis local contrast enhancement
    if rng.random() < mantis_p:
        strength = float(rng.uniform(0.3, 0.7))
        applied.append(f"mantis(strength={strength:.3f})")
        mantis = MantisLocalContrast(kernel_sigma=15.0, strength=strength)
        image, _ = mantis(image, heatmap)

    # Gaussian blur — VERY mild to avoid destroying 1px particles
    if rng.random() < blur_p:
        applied.append("blur(0.3-0.8)")
        blur = GaussianBlur(sigma_range=(0.3, 0.8))
        image, _ = blur(image, heatmap, rng)

    # Gamma correction (beam intensity, detector response)
    if rng.random() < gamma_p:
        # gamma itself is sampled inside GammaCorrection; we can approximate by sampling here for logging clarity
        applied.append("gamma")
        gamma = GammaCorrection()
        image, _ = gamma(image, heatmap, rng)

    # Brightness/contrast (gain/amplifier variation)
    if rng.random() < brightness_contrast_p:
        applied.append("brightness_contrast")
        bc = BrightnessContrast()
        image, _ = bc(image, heatmap, rng)

    # Gaussian noise (detector shot noise)
    if rng.random() < noise_p:
        applied.append("gaussian_noise")
        noise = GaussianNoise()
        image, _ = noise(image, heatmap, rng)

    # Salt & pepper noise (hot pixels, cosmic rays)
    if rng.random() < salt_pepper_p:
        applied.append("salt_pepper")
        sp = SaltPepperNoise()
        image, _ = sp(image, heatmap, rng)

    # Cutout — only zero the image, NOT the heatmap (erasing heatmap
    # teaches the network that particles don't exist where they do)
    if rng.random() < cutout_p:
        applied.append("cutout")
        cutout = Cutout(size_frac=1.0/20.0, max_count=1)
        image_out, _ = cutout(image, heatmap, rng)
        image = image_out

    # Final robust normalization step to avoid washed-out patches.
    # Triggered only when image is near-saturated or low-contrast.
    cur_mean = float(np.asarray(image, dtype=np.float32).mean())
    cur_std = float(np.asarray(image, dtype=np.float32).std())
    # Trigger stabilization for both severe saturation and subtle washed-out cases
    # observed in logs (mean ~0.95 with std ~0.026).
    if (cur_mean > 0.97) or (cur_std < 0.02) or (cur_mean > 0.93 and cur_std < 0.03):
        image_before_norm = np.asarray(image, dtype=np.float32).copy()
        # Use tighter percentiles to avoid pushing large regions to exact 0/1.
        norm = NormalizeRobust(p_lo=5.0, p_hi=95.0)
        image, _ = norm(image, heatmap)
        # If still flattened after normalization, blend back original local texture.
        post_mean = float(np.asarray(image, dtype=np.float32).mean())
        post_std = float(np.asarray(image, dtype=np.float32).std())
        if post_mean > 0.95 and post_std < 0.02:
            image = np.clip(0.6 * image + 0.4 * image_before_norm, 0.0, 1.0).astype(np.float32)
            applied.append("texture_restore_blend")
        applied.append("normalize_robust(5-95)")

    return image, heatmap


if __name__ == "__main__":
    # Quick self-test
    print("Testing augmentations...")

    # Create synthetic data
    h, w = 512, 512
    image = np.random.uniform(0.3, 0.7, size=(3, h, w)).astype(np.float32)
    heatmap = np.zeros((2, h, w), dtype=np.float32)

    # Add a few Gaussian peaks to heatmap for testing
    from scipy.stats import multivariate_normal
    y, x = np.mgrid[0:h, 0:w]
    pos = np.dstack((y, x))

    for cy, cx in [(100, 100), (300, 300), (400, 200)]:
        rv = multivariate_normal([cy, cx], [[25, 0], [0, 25]])
        heatmap[0] += rv.pdf(pos) * 10

    heatmap = np.clip(heatmap, 0, 1)

    rng = np.random.default_rng(42)

    # Test each augmentation
    print("  - ElasticDeform...", end=" ")
    elastic = ElasticDeform()
    img_aug, hm_aug = elastic(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - GaussianNoise...", end=" ")
    noise = GaussianNoise()
    img_aug, hm_aug = noise(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - CLAHEPreprocess...", end=" ")
    clahe = CLAHEPreprocess()
    img_aug, hm_aug = clahe(image.copy(), heatmap.copy())
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - GammaCorrection...", end=" ")
    gamma = GammaCorrection()
    img_aug, hm_aug = gamma(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - SaltPepperNoise...", end=" ")
    sp = SaltPepperNoise()
    img_aug, hm_aug = sp(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - RandomErasing...", end=" ")
    erasing = RandomErasing()
    img_aug, hm_aug = erasing(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - Cutout...", end=" ")
    cutout = Cutout()
    img_aug, hm_aug = cutout(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - GaussianBlur...", end=" ")
    blur = GaussianBlur()
    img_aug, hm_aug = blur(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("  - Full pipeline...", end=" ")
    img_aug, hm_aug = apply_augmentation(image.copy(), heatmap.copy(), rng)
    assert img_aug.shape == image.shape and hm_aug.shape == heatmap.shape
    print("OK")

    print("\nAll tests passed!")
