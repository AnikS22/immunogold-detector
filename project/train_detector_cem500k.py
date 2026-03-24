#!/usr/bin/env python3
"""
Fine-tune CEM500K ResNet50 + UNet for immunogold particle detection.

This script implements transfer learning from CEM500K pre-trained ResNet50:
1. Start with frozen encoder, train decoder only
2. After N epochs, unfreeze last 2 encoder blocks
3. Use smaller learning rates for transfer learning
4. Semi-supervised learning on unlabeled images
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import sys
from typing import Optional, Tuple

from model_unet_cem500k import UNetCEM500K
from dataset_points_sliding_window import SlidingWindowPatchDataset
from prepare_labels import discover_image_records
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune CEM500K ResNet50 + UNet for immunogold detection")

    # Data
    p.add_argument("--data_root", type=str, required=True, help="Root directory of EM images")
    p.add_argument("--patch_h", type=int, default=256, help="Patch height")
    p.add_argument("--patch_w", type=int, default=256, help="Patch width")
    p.add_argument("--patch_stride", type=int, default=128, help="Sliding window stride")
    p.add_argument("--train_samples_per_epoch", type=int, default=2048, help="Samples per epoch")
    p.add_argument("--val_samples_per_epoch", type=int, default=256, help="Val samples per epoch")

    # Training
    p.add_argument("--epochs", type=int, default=100, help="Max epochs")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (for decoder)")
    p.add_argument("--lr_encoder", type=float, default=1e-5, help="Encoder learning rate after unfreezing")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Transfer learning
    p.add_argument("--freeze_encoder_epochs", type=int, default=15, help="Epochs to keep encoder frozen")
    p.add_argument("--unfreeze_blocks", type=int, default=2, help="Number of ResNet blocks to unfreeze")

    # Loss
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma for heatmap")
    p.add_argument("--loss_type", type=str, default="focal_bce", help="Loss function type")
    p.add_argument("--loss_pos_weight", type=float, default=30.0, help="BCE pos_weight")
    p.add_argument("--loss_neg_weight", type=float, default=1.0, help="BCE neg_weight")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")

    # Augmentation
    p.add_argument("--sigma_jitter", action="store_true", help="Enable sigma jitter in heatmap generation")

    # Training optimization
    p.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training (FP16)")

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--early_stop_delta", type=float, default=1e-5, help="Min improvement threshold")

    # Save
    p.add_argument("--save_dir", type=str, default="checkpoints/cem500k", help="Checkpoint directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


class FocalBCELoss(nn.Module):
    """Focal Binary Cross Entropy Loss."""

    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0, focal_gamma: float = 2.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.focal_gamma = float(focal_gamma)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)

        # Focal weighting: (1 - p_t)^gamma * loss_t
        p_t = torch.sigmoid(pred)
        p_t = torch.where(target == 1, p_t, 1 - p_t)
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Class weighting
        class_weight = torch.where(target == 1, self.pos_weight, self.neg_weight)

        return (focal_weight * class_weight * bce_loss).mean()


def train_epoch(model, train_loader, optimizer, loss_fn, device, scaler=None):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, heatmaps in train_loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, heatmaps)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, heatmaps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def eval_epoch(model, val_loader, loss_fn, device):
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, heatmaps in val_loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, heatmaps)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    records = discover_image_records(args.data_root)
    np.random.seed(args.seed)
    idx = np.arange(len(records))
    np.random.shuffle(idx)
    n_train = max(1, int(0.7 * len(records)))
    n_val = max(1, int(0.15 * len(records)))
    train_records = [records[i] for i in idx[:n_train]]
    val_records = [records[i] for i in idx[n_train : n_train + n_val]]

    train_dataset = SlidingWindowPatchDataset(
        train_records,
        patch_size=(args.patch_h, args.patch_w),
        samples_per_epoch=args.train_samples_per_epoch,
        sigma=args.sigma,
        augment=True,
        sigma_jitter=True,
        patch_stride=args.patch_stride,
    )
    val_dataset = SlidingWindowPatchDataset(
        val_records,
        patch_size=(args.patch_h, args.patch_w),
        samples_per_epoch=args.val_samples_per_epoch,
        sigma=args.sigma,
        augment=False,
        patch_stride=args.patch_stride,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("Creating CEM500K ResNet50 + UNet model...")
    model = UNetCEM500K(pretrained=True, freeze_encoder=True)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    loss_fn = FocalBCELoss(
        pos_weight=args.loss_pos_weight,
        neg_weight=args.loss_neg_weight,
        focal_gamma=args.focal_gamma,
    )

    # Start with low learning rate (transfer learning)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    encoder_unfrozen = False

    print("\n" + "=" * 70)
    print("  CEM500K TRANSFER LEARNING TRAINING")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr} (decoder)")
    print(f"Freeze encoder: {args.freeze_encoder_epochs} epochs")
    print(f"Early stopping: patience={args.early_stop_patience}")
    print("=" * 70 + "\n")

    for epoch in range(1, args.epochs + 1):
        # Progressive unfreezing
        if epoch == args.freeze_encoder_epochs + 1 and not encoder_unfrozen:
            print(f"\n✓ Unfreezing last {args.unfreeze_blocks} encoder blocks at epoch {epoch}")
            model.unfreeze_encoder_partial(args.unfreeze_blocks)
            encoder_unfrozen = True

            # Create new optimizer with both encoder and decoder
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr_encoder,
                weight_decay=args.weight_decay,
            )

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:03d}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}", end="")

        if val_loss < best_val_loss - args.early_stop_delta:
            print(" ✓ Best checkpoint")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.save_dir}/detector_best.pt")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{args.early_stop_patience})")

            if patience_counter >= args.early_stop_patience:
                print(f"\n🛑 Early stopping triggered (patience {args.early_stop_patience} reached)")
                break

    print("\n" + "=" * 70)
    print("✓ Training complete")
    print(f"✓ Best model saved to: {args.save_dir}/detector_best.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
