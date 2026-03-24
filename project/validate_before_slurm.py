#!/usr/bin/env python3
"""
Fast local checks before sbatch — catches import/shape/loss crashes without the cluster.

Usage (from `project/`):
  python validate_before_slurm.py
  python validate_before_slurm.py --data-root "data/Max Planck Data/Gold Particle Labelling/analyzed synapses"

Exit code 0 = all checks passed; non-zero = fix before submitting Slurm.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}", file=sys.stderr)


def check_imports() -> bool:
    print("\n[1/5] Imports (torch, timm, training modules)")
    try:
        import torch  # noqa: F401

        _ok("torch")
    except ImportError as e:
        _fail(f"torch: {e}")
        return False
    try:
        import timm  # noqa: F401

        _ok("timm")
    except ImportError as e:
        _fail(f"timm (pip install timm): {e}")
        return False
    try:
        from model_centernet_cem500k import CenterNetCEM500K
        from loss_functions_advanced import CenterNetAdvancedLoss
        from dataset_centernet import create_dataloaders, CenterNetParticleDataset, discover_image_records

        _ok("model_centernet_cem500k, loss_functions_advanced, dataset_centernet")
    except ImportError as e:
        _fail(f"project modules: {e}")
        return False
    return True


def check_synthetic_forward_backward(device: str) -> bool:
    print(f"\n[2/5] Synthetic batch: forward + loss (FP32) + backward on {device}")
    import torch
    from torch.cuda.amp import autocast, GradScaler

    from model_centernet_cem500k import CenterNetCEM500K
    from loss_functions_advanced import CenterNetAdvancedLoss

    torch.manual_seed(0)
    dev = torch.device(device)
    model = CenterNetCEM500K(pretrained=False, freeze_encoder=True).to(dev)
    model.train()
    crit = CenterNetAdvancedLoss(
        class_weights={0: 1.0, 1: 11.0},
        label_smoothing=0.05,
    )
    b, h, w = 2, 256, 256
    oh, ow = h // 4, w // 4
    x = torch.randn(b, 3, h, w, device=dev)
    targets = {
        "centers": torch.rand(b, 1, oh, ow, device=dev) * 0.1,
        "class_ids": torch.zeros(b, oh, ow, dtype=torch.long, device=dev),
        "sizes": torch.zeros(b, 2, oh, ow, device=dev),
        "offsets": torch.zeros(b, 2, oh, ow, device=dev),
        "confidence": torch.zeros(b, 1, oh, ow, device=dev),
    }
    targets["class_ids"][:, 8, 8] = 1
    targets["centers"][:, 0, 8, 8] = 1.0

    use_amp = device == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler() if use_amp else None

    opt.zero_grad(set_to_none=True)
    if use_amp:
        with autocast():
            pred = model(x)
        pred = {k: v.float() for k, v in pred.items()}
        loss = crit(pred, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        pred = model(x)
        pred = {k: v.float() for k, v in pred.items()}
        loss = crit(pred, targets)
        loss.backward()
        opt.step()

    if not torch.isfinite(loss).item():
        _fail(f"loss is not finite: {loss.item()}")
        return False
    _ok(f"loss={loss.item():.6f} (finite)")
    return True


def check_py_compile() -> bool:
    print("\n[3/5] Python bytecode compile (syntax + import structure)")
    import py_compile

    project = Path(__file__).resolve().parent
    files = [
        "train_centernet.py",
        "train_centernet_enhanced.py",
        "loss_functions_advanced.py",
        "model_centernet_cem500k.py",
        "dataset_centernet.py",
        "evaluate_detector.py",
    ]
    ok = True
    for name in files:
        path = project / name
        if not path.is_file():
            _fail(f"missing {name}")
            ok = False
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            _ok(name)
        except py_compile.PyCompileError as e:
            _fail(f"{name}: {e}")
            ok = False
    return ok


def check_data_sample(data_root: str | None, device: str) -> bool:
    print("\n[4/5] Optional real data (one batch, num_workers=0)")
    if not data_root:
        print("  SKIP (pass --data-root to test TIFF/CSV loading)")
        return True
    root = Path(data_root)
    if not root.is_dir():
        _fail(f"data root not found: {root}")
        return False

    import torch
    from torch.utils.data import DataLoader

    from dataset_centernet import create_dataloaders, discover_image_records, CenterNetParticleDataset

    names = ["S1", "S4", "S7", "S8", "S13", "S15", "S22"]
    train = [n for n in names if (root / n).is_dir()]
    if len(train) == 0:
        _fail(f"no expected synapse folders under {root}")
        return False
    val = train[: min(2, len(train))]

    try:
        tl, vl = create_dataloaders(
            str(root),
            train,
            val,
            batch_size=2,
            num_workers=0,
            patch_size=256,
            patch_stride=128,
            sigma=1.0,
        )
        batch = next(iter(tl))
        _ok(f"CenterNetDataset / create_dataloaders: batch image shape {batch[0].shape}")
    except Exception as e:
        _fail(f"create_dataloaders: {e}")
        return False

    try:
        recs = discover_image_records(str(root))
        if not recs:
            _fail("discover_image_records returned empty")
            return False
        ds = CenterNetParticleDataset(recs[:3], samples_per_epoch=4, patch_size=256)
        img, tgt = ds[0]
        dev = torch.device(device)
        from model_centernet_cem500k import CenterNetCEM500K
        from loss_functions_advanced import CenterNetAdvancedLoss

        m = CenterNetCEM500K(pretrained=False, freeze_encoder=True).to(dev)
        crit = CenterNetAdvancedLoss(label_smoothing=0.05)
        img = img.unsqueeze(0).to(dev)
        tgt = {k: v.unsqueeze(0).to(dev) for k, v in tgt.items()}
        pred = m(img)
        pred = {k: v.float() for k, v in pred.items()}
        loss = crit(pred, tgt)
        if not torch.isfinite(loss):
            _fail(f"CenterNetParticleDataset loss not finite: {loss}")
            return False
        _ok(f"CenterNetParticleDataset + model + loss: {loss.item():.6f}")
    except Exception as e:
        _fail(f"particle dataset path: {e}")
        return False

    return True


def check_slurm_syntax(project: Path) -> bool:
    print("\n[5/5] Slurm/bash syntax (bash -n)")
    slurm_dir = project / "hpc"
    if not slurm_dir.is_dir():
        print("  SKIP no hpc/")
        return True
    ok = True
    for p in sorted(slurm_dir.glob("*.slurm")) + sorted(slurm_dir.glob("*.sh")):
        r = subprocess.run(
            ["bash", "-n", str(p)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            _fail(f"{p.name}: {r.stderr or r.stdout}")
            ok = False
        else:
            _ok(f"{p.name}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate project before Slurm submission")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional path to analyzed synapses (tests real TIFF/CSV pipeline)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="Device for synthetic test (default: cuda if available else cpu)",
    )
    args = parser.parse_args()
    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    project = Path(__file__).resolve().parent
    print("=" * 60)
    print("  validate_before_slurm.py")
    print(f"  project: {project}")
    print(f"  device:  {args.device}")
    print("=" * 60)

    # Short-circuit: do not run later checks if imports fail (avoids obscure tracebacks).
    checks = [
        ("imports", check_imports),
        ("synthetic forward/backward", lambda: check_synthetic_forward_backward(args.device)),
        ("py_compile", check_py_compile),
        ("optional data", lambda: check_data_sample(args.data_root, args.device)),
        ("slurm bash -n", lambda: check_slurm_syntax(project)),
    ]
    steps: list[bool] = []
    for label, fn in checks:
        ok = fn()
        steps.append(ok)
        if not ok:
            print(f"\nStopped after failed step: {label}", file=sys.stderr)
            break
    if steps and all(steps):
        print("\n" + "=" * 60)
        print("  All checks passed — safe to sync and sbatch.")
        print("=" * 60 + "\n")
        return 0
    print("\n" + "=" * 60, file=sys.stderr)
    print("  Fix failures above before submitting Slurm jobs.", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
