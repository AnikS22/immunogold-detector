"""Training script for CenterNet particle detector with CEM500K transfer learning."""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler
import logging

from model_centernet_cem500k import CenterNetCEM500K
from dataset_centernet import create_dataloaders
from loss_functions_advanced import CenterNetAdvancedLoss


def setup_logging(save_dir):
    """Setup logging to file and console."""
    log_file = Path(save_dir) / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_total_epochs):
    """Cosine annealing schedule with linear warmup."""

    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # Cosine decay
            progress = float(current_epoch - num_warmup_epochs) / float(
                max(1, num_total_epochs - num_warmup_epochs)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(optimizer, lr_lambda)


def apply_gradient_centralization(model):
    """Apply gradient centralization for training stability."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.add_(-param.grad.mean(dim=tuple(range(1, param.grad.dim())), keepdim=True))


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    use_amp=False,
    scaler=None,
    gradient_centralization=False,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = {key: val.to(device) for key, val in targets.items()}

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                predictions = model(images)
            # Loss in FP32 — focal + logs inside autocast (FP16) overflows to inf/nan.
            predictions = {k: v.float() for k, v in predictions.items()}
            loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()

            if gradient_centralization:
                apply_gradient_centralization(model)

            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: loss={avg_loss:.4f}")

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp: bool = False):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    for images, targets in val_loader:
        images = images.to(device)
        targets = {key: val.to(device) for key, val in targets.items()}

        if use_amp:
            with autocast():
                predictions = model(images)
            predictions = {k: v.float() for k, v in predictions.items()}
        else:
            predictions = model(images)
        loss = criterion(predictions, targets)
        total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train CenterNet particle detector")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--patch_h", type=int, default=256, help="Patch height")
    parser.add_argument("--patch_w", type=int, default=256, help="Patch width")
    parser.add_argument("--patch_stride", type=int, default=128, help="Patch stride")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_encoder", type=float, default=1e-5, help="Encoder learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=15, help="Epochs to freeze encoder")
    parser.add_argument("--unfreeze_blocks", type=int, default=2, help="ResNet blocks to unfreeze")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--early_stop_patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--early_stop_delta", type=float, default=1e-5, help="Early stopping delta")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints/centernet", help="Save directory")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--gradient_centralization", action="store_true", help="Use gradient centralization")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.save_dir)

    # Dataset
    logger.info("Loading dataset...")
    all_images = ["S1", "S4", "S7", "S8", "S13", "S15", "S22", "S25", "S27", "S29"]
    train_images = all_images[:7]
    val_images = all_images[7:9]

    train_loader, val_loader = create_dataloaders(
        args.data_root,
        train_images,
        val_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_h,
        patch_stride=args.patch_stride,
        sigma=args.sigma,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    logger.info("Creating model...")
    model = CenterNetCEM500K(freeze_encoder=True)
    model = model.to(device)

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    # Loss function
    criterion = CenterNetAdvancedLoss(
        center_weight=1.0,
        class_weight=1.0,
        size_weight=0.1,
        offset_weight=1.0,
        conf_weight=1.0,
        boundary_weight=0.1,
    )

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs
    )

    # Amp scaler
    scaler = GradScaler() if args.mixed_precision else None

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Phase 2: Unfreeze encoder
        if epoch == args.freeze_encoder_epochs:
            logger.info(f"Unfreezing encoder blocks at epoch {epoch}...")
            model.unfreeze_encoder(args.unfreeze_blocks)

            # Update optimizer with new parameter groups
            param_groups = model.get_parameter_groups(args.lr, args.lr_encoder)
            optimizer = Adam(param_groups, weight_decay=args.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs - epoch)

        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=args.mixed_precision,
            scaler=scaler,
            gradient_centralization=args.gradient_centralization,
        )

        val_loss = validate(model, val_loader, criterion, device, use_amp=args.mixed_precision)

        scheduler.step()

        logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss - args.early_stop_delta:
            best_val_loss = val_loss
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = Path(args.save_dir) / "detector_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info("Training complete")


if __name__ == "__main__":
    main()
