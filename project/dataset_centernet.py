"""CenterNet dataset loader for gold particle detection."""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import gaussian_filter
import tifffile
from typing import Dict, List, Tuple, Optional
from prepare_labels import _load_image_safe


class CenterNetDataset(Dataset):
    """CenterNet dataset for particle detection."""

    def __init__(
        self,
        data_root,
        image_names,
        patch_size=256,
        patch_stride=128,
        sigma=1.0,
        augment=False,
    ):
        """
        Args:
            data_root: Root directory with EM images and annotations
            image_names: List of image names (e.g., ['S1', 'S4', ...])
            patch_size: Size of patches (default 256)
            patch_stride: Stride for sliding window (default 128)
            sigma: Gaussian sigma for heatmap generation
            augment: Whether to apply augmentations
        """
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.sigma = sigma
        self.augment = augment

        # Pixel-per-micron conversion (from Max Planck data)
        self.pixels_per_micron = 1790.0  # verified from ImageJ ROI files

        # Discover patches
        self.patches = self._discover_patches(image_names)

    def _discover_patches(self, image_names):
        """Discover all patches from given images."""
        patches = []

        for img_name in image_names:
            img_dir = self.data_root / img_name
            if not img_dir.is_dir():
                print(f"Warning: Image folder not found: {img_dir}")
                continue

            # Same main-TIFF selection as __getitem__ (filenames are not always {img_name}.tif)
            tif_files = list(img_dir.glob("*.tif"))
            main_tif = None
            for tif in tif_files:
                name = tif.name.lower()
                if "color" not in name and "mask" not in name and "overlay" not in name:
                    main_tif = tif
                    break
            if main_tif is None and tif_files:
                main_tif = tif_files[0]
            if main_tif is None:
                print(f"Warning: No .tif in {img_dir}")
                continue

            img = _load_image_safe(str(main_tif))
            h, w = img.shape[:2]

            # Generate patches
            for y in range(0, h - self.patch_size + 1, self.patch_stride):
                for x in range(0, w - self.patch_size + 1, self.patch_stride):
                    patches.append((img_name, x, y))

        return patches

    def _load_annotations(self, img_name):
        """Load particle annotations from CSV files."""
        particles = {"6nm": [], "12nm": []}

        for size in ["6nm", "12nm"]:
            # Find annotation file (naming varies)
            result_dir = self.data_root / img_name / "Results"
            possible_names = [
                f"Results {size} XY in microns.csv",
                f"Results XY in microns {size}.csv",
            ]

            for fname in possible_names:
                csv_path = result_dir / fname
                if csv_path.exists():
                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            x_microns = float(row["X"])
                            y_microns = float(row["Y"])

                            # Convert to pixels
                            x_pixels = x_microns * self.pixels_per_micron
                            y_pixels = y_microns * self.pixels_per_micron

                            particles[size].append((x_pixels, y_pixels))
                    break

        return particles

    def _create_targets(self, patch_image, particles, patch_x, patch_y):
        """Create multi-task targets for CenterNet."""
        h, w = patch_image.shape[:2]
        out_h, out_w = h // 4, w // 4  # Output at 1/4 stride

        targets = {
            "centers": np.zeros((1, out_h, out_w), dtype=np.float32),
            "class_ids": np.zeros((out_h, out_w), dtype=np.int64),
            "sizes": np.zeros((2, out_h, out_w), dtype=np.float32),
            "offsets": np.zeros((2, out_h, out_w), dtype=np.float32),
            "confidence": np.zeros((1, out_h, out_w), dtype=np.float32),
        }

        # Place particles in patch
        for size, coords in particles.items():
            class_id = 0 if size == "6nm" else 1

            for x_pixel, y_pixel in coords:
                # Check if particle is in patch
                x_rel = x_pixel - patch_x
                y_rel = y_pixel - patch_y

                if 0 <= x_rel < w and 0 <= y_rel < h:
                    # Downsample to 1/4 stride
                    cy = int(y_rel / 4)
                    cx = int(x_rel / 4)

                    # Sub-pixel offset
                    offset_y = (y_rel / 4) - cy
                    offset_x = (x_rel / 4) - cx

                    # Ensure within bounds
                    if 0 <= cy < out_h and 0 <= cx < out_w:
                        # Center heatmap (Gaussian)
                        y_range = np.arange(out_h)
                        x_range = np.arange(out_w)
                        yy, xx = np.meshgrid(y_range, x_range, indexing="ij")
                        gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * self.sigma ** 2))
                        targets["centers"][0] = np.maximum(targets["centers"][0], gaussian)

                        # Class
                        targets["class_ids"][cy, cx] = class_id

                        # Size (particle radius in pixels)
                        radius = 3 if size == "6nm" else 6
                        targets["sizes"][0, cy, cx] = radius
                        targets["sizes"][1, cy, cx] = radius

                        # Offset
                        targets["offsets"][0, cy, cx] = offset_x
                        targets["offsets"][1, cy, cx] = offset_y

                        # Confidence (how sure we are about detection)
                        targets["confidence"][0, cy, cx] = 1.0

        return targets

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_name, patch_x, patch_y = self.patches[idx]

        # Load image (find first .tif file in directory)
        img_dir = self.data_root / img_name
        tif_files = list(img_dir.glob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in {img_dir}")
        # Find the main image (exclude color, mask, overlay)
        main_tif = None
        for tif in tif_files:
            name = tif.name.lower()
            if "color" not in name and "mask" not in name and "overlay" not in name:
                main_tif = tif
                break
        if main_tif is None:
            main_tif = tif_files[0]

        image = _load_image_safe(str(main_tif)).astype(np.float32)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Extract patch
        patch_image = image[
            patch_y : patch_y + self.patch_size,
            patch_x : patch_x + self.patch_size,
        ]

        # Load annotations
        particles = self._load_annotations(img_name)

        # Create targets
        targets = self._create_targets(patch_image, particles, patch_x, patch_y)

        # Convert to CHW float tensor (match CenterNetParticleDataset / timm)
        if patch_image.ndim == 2:
            image_tensor = torch.from_numpy(np.stack([patch_image] * 3, axis=0).astype(np.float32))
        elif patch_image.ndim == 3:
            c = patch_image.shape[2]
            if c == 1:
                g = patch_image[:, :, 0]
                image_tensor = torch.from_numpy(np.stack([g, g, g], axis=0).astype(np.float32))
            elif c >= 3:
                image_tensor = torch.from_numpy(
                    np.transpose(patch_image[:, :, :3], (2, 0, 1)).astype(np.float32)
                )
            else:
                raise ValueError(f"Unexpected channel count {c} in patch shape {patch_image.shape}")
        else:
            raise ValueError(f"Unexpected patch ndim {patch_image.ndim}")

        target_tensors = {
            key: torch.from_numpy(val) for key, val in targets.items()
        }

        return image_tensor, target_tensors


def create_dataloaders(
    data_root,
    train_images,
    val_images,
    batch_size=8,
    num_workers=4,
    **dataset_kwargs,
):
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader

    train_dataset = CenterNetDataset(
        data_root, train_images, augment=True, **dataset_kwargs
    )
    val_dataset = CenterNetDataset(
        data_root, val_images, augment=False, **dataset_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def discover_image_records(data_root: str) -> List[Dict]:
    """Discover all image records (image path + annotations)."""
    data_root = Path(data_root)
    records = []

    # Expected image names
    image_names = ["S1", "S4", "S7", "S8", "S13", "S15", "S22", "S25", "S27", "S29"]

    for img_name in image_names:
        img_dir = data_root / img_name
        if img_dir.exists():
            records.append({
                "image_name": img_name,
                "image_dir": str(img_dir),
                "data_root": str(data_root)  # Keep the actual analyzed synapses root
            })

    return records


class CenterNetParticleDataset(Dataset):
    """Random patch sampling dataset for CenterNet (compatible with enhanced training)."""

    def __init__(
        self,
        records: List[Dict],
        patch_size: int = 256,
        sigma: float = 1.0,
        samples_per_epoch: int = 2048,
    ):
        """
        Args:
            records: List of image records from discover_image_records()
            patch_size: Size of patches to extract
            sigma: Gaussian sigma for heatmap generation
            samples_per_epoch: Number of samples to generate per epoch
        """
        self.records = records
        self.patch_size = patch_size
        self.sigma = sigma
        self.samples_per_epoch = samples_per_epoch
        self.data_root = Path(records[0]["data_root"]) if records else Path(".")

        # Pixel-per-micron conversion
        self.pixels_per_micron = 1790.0  # verified from ImageJ ROI files

    def _load_image(self, img_name: str, data_root: Path) -> np.ndarray:
        """Load and normalize image."""
        img_dir = data_root / img_name
        tif_files = list(img_dir.glob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in {img_dir}")

        # Find main image (exclude color, mask, overlay)
        main_tif = None
        for tif in tif_files:
            name = tif.name.lower()
            if "color" not in name and "mask" not in name and "overlay" not in name:
                main_tif = tif
                break
        if main_tif is None:
            main_tif = tif_files[0]

        image = _load_image_safe(str(main_tif)).astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def _load_annotations(self, img_name: str, data_root: Path) -> Dict[str, List[Tuple[float, float]]]:
        """Load particle annotations."""
        particles = {"6nm": [], "12nm": []}

        for size in ["6nm", "12nm"]:
            result_dir = data_root / img_name / "Results"
            possible_names = [
                f"Results {size} XY in microns.csv",
                f"Results XY in microns {size}.csv",
            ]

            for fname in possible_names:
                csv_path = result_dir / fname
                if csv_path.exists():
                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                x_microns = float(row["X"])
                                y_microns = float(row["Y"])
                                x_pixels = x_microns * self.pixels_per_micron
                                y_pixels = y_microns * self.pixels_per_micron
                                particles[size].append((x_pixels, y_pixels))
                            except (ValueError, KeyError):
                                pass
                    break

        return particles

    def _create_targets(
        self, patch_image: np.ndarray, particles: Dict, patch_x: int, patch_y: int
    ) -> Dict:
        """Create multi-task targets."""
        h, w = patch_image.shape[:2]
        out_h, out_w = h // 4, w // 4

        targets = {
            "centers": np.zeros((1, out_h, out_w), dtype=np.float32),
            "class_ids": np.zeros((out_h, out_w), dtype=np.int64),
            "sizes": np.zeros((2, out_h, out_w), dtype=np.float32),
            "offsets": np.zeros((2, out_h, out_w), dtype=np.float32),
            "confidence": np.zeros((1, out_h, out_w), dtype=np.float32),
        }

        for size, coords in particles.items():
            class_id = 0 if size == "6nm" else 1

            for x_pixel, y_pixel in coords:
                x_rel = x_pixel - patch_x
                y_rel = y_pixel - patch_y

                if 0 <= x_rel < w and 0 <= y_rel < h:
                    cy = int(y_rel / 4)
                    cx = int(x_rel / 4)

                    offset_y = (y_rel / 4) - cy
                    offset_x = (x_rel / 4) - cx

                    if 0 <= cy < out_h and 0 <= cx < out_w:
                        y_range = np.arange(out_h)
                        x_range = np.arange(out_w)
                        yy, xx = np.meshgrid(y_range, x_range, indexing="ij")
                        gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * self.sigma ** 2))
                        targets["centers"][0] = np.maximum(targets["centers"][0], gaussian)

                        targets["class_ids"][cy, cx] = class_id
                        radius = 3 if size == "6nm" else 6
                        targets["sizes"][0, cy, cx] = radius
                        targets["sizes"][1, cy, cx] = radius
                        targets["offsets"][0, cy, cx] = offset_x
                        targets["offsets"][1, cy, cx] = offset_y
                        targets["confidence"][0, cy, cx] = 1.0

        return targets

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """Return random patch from random image."""
        # Get random image record
        record = self.records[np.random.randint(len(self.records))]
        img_name = record["image_name"]
        img_dir = Path(record["image_dir"])

        # Load image directly from image_dir
        tif_files = list(img_dir.glob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in {img_dir}")

        # Find main image (exclude color, mask, overlay)
        main_tif = None
        for tif in tif_files:
            name = tif.name.lower()
            if "color" not in name and "mask" not in name and "overlay" not in name:
                main_tif = tif
                break
        if main_tif is None:
            main_tif = tif_files[0]

        image = _load_image_safe(str(main_tif)).astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        h, w = image.shape[:2]

        # Random patch location
        patch_x = np.random.randint(0, max(1, w - self.patch_size + 1))
        patch_y = np.random.randint(0, max(1, h - self.patch_size + 1))

        # Extract patch
        patch_image = image[patch_y : patch_y + self.patch_size, patch_x : patch_x + self.patch_size]

        # Ensure correct size (pad if necessary)
        if patch_image.shape[:2] != (self.patch_size, self.patch_size):
            if len(patch_image.shape) == 3:
                # 3-channel image
                padded = np.zeros((self.patch_size, self.patch_size, patch_image.shape[2]), dtype=np.float32)
            else:
                # Grayscale image
                padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            padded[: patch_image.shape[0], : patch_image.shape[1]] = patch_image
            patch_image = padded

        # Load annotations and create targets
        # The data_root stored in record points to analyzed synapses parent
        data_root_for_annotations = Path(record["data_root"])
        particles = self._load_annotations(img_name, data_root_for_annotations)
        targets = self._create_targets(patch_image, particles, patch_x, patch_y)

        # Convert to tensors
        if len(patch_image.shape) == 2:
            # Grayscale - convert to 3-channel
            image_tensor = torch.from_numpy(np.stack([patch_image]*3, axis=0).astype(np.float32))
        else:
            # Already 3-channel - transpose to CHW format
            image_tensor = torch.from_numpy(np.transpose(patch_image, (2, 0, 1)).astype(np.float32))

        target_tensors = {key: torch.from_numpy(val) for key, val in targets.items()}

        return image_tensor, target_tensors


if __name__ == "__main__":
    # Test dataset
    data_root = "data/Max Planck Data/Gold Particle Labelling/analyzed synapses"
    dataset = CenterNetDataset(data_root, ["S1"], patch_size=256, patch_stride=128)
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        img, targets = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Target shapes:")
        for key, val in targets.items():
            print(f"  {key}: {val.shape}")
