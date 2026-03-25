#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch

from model_centernet_cem500k import CenterNetCEM500K
from prepare_labels import discover_image_records, _load_image_safe


def estimate_bottom_black_bar_px(
    image: np.ndarray,
    dark_threshold: float = 8.0,
    min_dark_fraction: float = 0.98,
) -> int:
    """
    Estimate thickness of a dark bottom bar in pixels.
    Uses row-wise dark-pixel fraction from bottom upwards.
    """
    if image.ndim == 3:
        # Use luminance proxy; channels in TIFF may vary.
        gray = image[..., 0].astype(np.float32)
    else:
        gray = image.astype(np.float32)
    if gray.size == 0:
        return 0
    # Convert threshold to image scale.
    # If image looks 0..1, dark threshold is interpreted in that scale.
    thr = dark_threshold
    if float(gray.max()) <= 1.5:
        thr = dark_threshold / 255.0

    h = gray.shape[0]
    bar = 0
    for y in range(h - 1, -1, -1):
        frac_dark = float(np.mean(gray[y] <= thr))
        if frac_dark >= float(min_dark_fraction):
            bar += 1
        else:
            break
    return int(bar)


def image_to_chw_01(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image.astype(np.float32)
    mn, mx = float(image.min()), float(image.max())
    if mx > mn:
        image = (image - mn) / (mx - mn)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return np.transpose(image[:, :, :3], (2, 0, 1))


def tiled_predict(
    model: torch.nn.Module,
    chw: np.ndarray,
    device: torch.device,
    tile: int,
    stride: int,
) -> Dict[str, np.ndarray]:
    _, h, w = chw.shape
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if not ys or ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    if not xs or xs[-1] != w - tile:
        xs.append(max(0, w - tile))

    out = {
        "centers": np.zeros((h // 4, w // 4), np.float32),
        "class0": np.zeros((h // 4, w // 4), np.float32),
        "class1": np.zeros((h // 4, w // 4), np.float32),
        "offx": np.zeros((h // 4, w // 4), np.float32),
        "offy": np.zeros((h // 4, w // 4), np.float32),
        "conf": np.zeros((h // 4, w // 4), np.float32),
    }
    cnt = np.zeros((h // 4, w // 4), np.float32)

    model.eval()
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = chw[:, y0 : y0 + tile, x0 : x0 + tile]
                t = torch.from_numpy(patch[None]).float().to(device)
                pred = model(t)

                p_cent = torch.sigmoid(pred["centers"])[0, 0].cpu().numpy()
                p_cls = pred["classes"][0].cpu().numpy()
                p_off = pred["offsets"][0].cpu().numpy()
                p_conf = torch.sigmoid(pred["confidence"])[0, 0].cpu().numpy()

                yy0, xx0 = y0 // 4, x0 // 4
                yy1, xx1 = yy0 + p_cent.shape[0], xx0 + p_cent.shape[1]
                out["centers"][yy0:yy1, xx0:xx1] += p_cent
                out["class0"][yy0:yy1, xx0:xx1] += p_cls[0]
                out["class1"][yy0:yy1, xx0:xx1] += p_cls[1]
                out["offx"][yy0:yy1, xx0:xx1] += p_off[0]
                out["offy"][yy0:yy1, xx0:xx1] += p_off[1]
                out["conf"][yy0:yy1, xx0:xx1] += p_conf
                cnt[yy0:yy1, xx0:xx1] += 1.0

    cnt = np.maximum(cnt, 1e-6)
    for k in out:
        out[k] /= cnt
    return out


def decode_detections(
    out: Dict[str, np.ndarray],
    pre_threshold: float,
    min_distance: int,
    max_det_per_class: int,
    ignore_bottom_px: int = 0,
) -> List[Tuple[float, float, int, float]]:
    score = out["centers"] * out["conf"]
    if int(ignore_bottom_px) > 0:
        h4 = score.shape[0]
        cut_h4 = max(0, h4 - (int(ignore_bottom_px) // 4))
        if cut_h4 < h4:
            score[cut_h4:, :] = 0.0
    ys, xs = np.where(score >= float(pre_threshold))
    if len(xs) == 0:
        return []

    conf = score[ys, xs]
    cls = (out["class1"][ys, xs] > out["class0"][ys, xs]).astype(np.int64)
    ox = out["offx"][ys, xs]
    oy = out["offy"][ys, xs]

    dets = [
        ((xs[i] + ox[i]) * 4.0, (ys[i] + oy[i]) * 4.0, int(cls[i]), float(conf[i]))
        for i in range(len(xs))
    ]

    final: List[Tuple[float, float, int, float]] = []
    r2 = float(min_distance * min_distance)
    for k in [0, 1]:
        pool = [d for d in dets if d[2] == k]
        pool.sort(key=lambda z: z[3], reverse=True)
        kept: List[Tuple[float, float, int, float]] = []
        for d in pool:
            ok = True
            for q in kept:
                dx = d[0] - q[0]
                dy = d[1] - q[1]
                if dx * dx + dy * dy < r2:
                    ok = False
                    break
            if ok:
                kept.append(d)
            if len(kept) >= max_det_per_class:
                break
        final.extend(kept)
    return final


def main() -> None:
    p = argparse.ArgumentParser(description="CenterNet inference to prediction CSV for evaluation.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--pre_threshold", type=float, default=0.02)
    p.add_argument("--min_distance", type=int, default=4)
    p.add_argument("--max_det_per_class", type=int, default=500)
    p.add_argument(
        "--ignore_bottom_px",
        type=int,
        default=0,
        help="Suppress detections in bottom N pixels (useful for dark bars).",
    )
    p.add_argument(
        "--auto_ignore_bottom_bar",
        action="store_true",
        help="Automatically detect dark bottom bar and suppress detections there.",
    )
    p.add_argument("--bar_dark_threshold", type=float, default=8.0)
    p.add_argument("--bar_min_dark_fraction", type=float, default=0.98)
    args = p.parse_args()

    records = discover_image_records(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNetCEM500K(pretrained=False, freeze_encoder=True).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "x", "y", "class_id", "confidence"])
        for rec in records:
            img = _load_image_safe(rec.image_path)
            chw = image_to_chw_01(img)
            ignore_bottom_px = int(args.ignore_bottom_px)
            if args.auto_ignore_bottom_bar:
                ignore_bottom_px = max(
                    ignore_bottom_px,
                    estimate_bottom_black_bar_px(
                        img,
                        dark_threshold=float(args.bar_dark_threshold),
                        min_dark_fraction=float(args.bar_min_dark_fraction),
                    ),
                )
            out = tiled_predict(
                model=model,
                chw=chw,
                device=device,
                tile=int(args.tile),
                stride=int(args.stride),
            )
            dets = decode_detections(
                out=out,
                pre_threshold=float(args.pre_threshold),
                min_distance=int(args.min_distance),
                max_det_per_class=int(args.max_det_per_class),
                ignore_bottom_px=ignore_bottom_px,
            )
            for x, y, cls, conf in dets:
                w.writerow([rec.image_id, f"{x:.3f}", f"{y:.3f}", int(cls), f"{conf:.6f}"])

    print(f"Wrote predictions: {out_csv}")


if __name__ == "__main__":
    main()
