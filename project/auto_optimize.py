"""
Auto-optimizer: analyze eval results mid-job and run additional training rounds
with adjusted hyperparameters to push F1 higher.

Called from Slurm after initial train → infer → eval cycle.
Reads eval_results.txt, diagnoses the failure mode, adjusts strategy, retrains.
Repeats up to MAX_ROUNDS or until F1 target is hit.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


TARGET_F1 = 0.91
MAX_ROUNDS = 4  # max optimization rounds after initial training


def parse_eval_results(eval_path):
    """Parse eval_results.txt to extract best metrics."""
    best_macro_f1 = 0.0
    best_threshold = 0.0
    best_metrics = {}

    with open(eval_path) as f:
        lines = f.readlines()

    current_threshold = None
    current_block = {}

    for line in lines:
        line = line.strip()

        m = re.match(r'best_threshold=([\d.]+)\s+best_macro_f1=([\d.]+)', line)
        if m:
            best_threshold = float(m.group(1))
            best_macro_f1 = float(m.group(2))

        m = re.match(r'threshold=([\d.]+)', line)
        if m:
            current_threshold = float(m.group(1))

        for key in ['all.f1', 'all.precision', 'all.recall', 'all.tp', 'all.fp', 'all.fn',
                     'class_6nm.f1', 'class_6nm.precision', 'class_6nm.recall',
                     'class_6nm.tp', 'class_6nm.fp', 'class_6nm.fn',
                     'class_12nm.f1', 'class_12nm.precision', 'class_12nm.recall',
                     'class_12nm.tp', 'class_12nm.fp', 'class_12nm.fn',
                     'macro.f1']:
            m = re.match(rf'{re.escape(key)}=([\d.nan]+)', line)
            if m:
                try:
                    current_block[key] = float(m.group(1))
                except ValueError:
                    current_block[key] = 0.0

        if 'macro.f1' in current_block and current_threshold is not None:
            if current_block.get('macro.f1', 0) > best_macro_f1:
                best_macro_f1 = current_block['macro.f1']
                best_threshold = current_threshold
                best_metrics = dict(current_block)
            current_block = {}

    return {
        'best_macro_f1': best_macro_f1,
        'best_threshold': best_threshold,
        'metrics': best_metrics,
    }


def diagnose(results):
    """Diagnose the failure mode from eval results.

    Returns a list of diagnoses and recommended adjustments.
    """
    m = results['metrics']
    f1 = results['best_macro_f1']
    adjustments = []

    if not m:
        adjustments.append(('catastrophic', 'Model produced no valid detections'))
        return adjustments

    all_prec = m.get('all.precision', 0)
    all_rec = m.get('all.recall', 0)
    all_tp = m.get('all.tp', 0)
    all_fp = m.get('all.fp', 0)
    all_fn = m.get('all.fn', 0)

    f1_6nm = m.get('class_6nm.f1', 0)
    f1_12nm = m.get('class_12nm.f1', 0)
    rec_6nm = m.get('class_6nm.recall', 0)
    rec_12nm = m.get('class_12nm.recall', 0)
    prec_6nm = m.get('class_6nm.precision', 0)
    prec_12nm = m.get('class_12nm.precision', 0)

    # Diagnosis 1: FP flood (low precision)
    if all_prec < 0.3 and all_rec > 0.3:
        adjustments.append(('fp_flood', f'Precision={all_prec:.3f} too low, FP={int(all_fp)}'))

    # Diagnosis 2: Under-detection (low recall)
    if all_rec < 0.3:
        adjustments.append(('low_recall', f'Recall={all_rec:.3f}, missing {int(all_fn)} particles'))

    # Diagnosis 3: 12nm class failure
    if f1_12nm < 0.1 and f1_6nm > 0.2:
        adjustments.append(('12nm_weak', f'12nm F1={f1_12nm:.3f} vs 6nm F1={f1_6nm:.3f}'))

    # Diagnosis 4: Both classes weak
    if f1_6nm < 0.2 and f1_12nm < 0.2:
        adjustments.append(('both_weak', f'6nm F1={f1_6nm:.3f}, 12nm F1={f1_12nm:.3f}'))

    # Diagnosis 5: Good recall but classification errors
    if all_rec > 0.5 and all_prec > 0.3 and f1 < TARGET_F1:
        if abs(f1_6nm - f1_12nm) > 0.3:
            adjustments.append(('class_imbalance', f'6nm/12nm gap: {f1_6nm:.3f}/{f1_12nm:.3f}'))

    # Diagnosis 6: Already decent, needs fine-tuning
    if f1 > 0.5:
        adjustments.append(('finetune', f'F1={f1:.3f}, needs refinement'))

    if not adjustments:
        adjustments.append(('unknown', f'F1={f1:.3f}, no clear diagnosis'))

    return adjustments


def generate_retry_args(base_args, diagnoses, round_num, prev_results):
    """Generate adjusted training arguments based on diagnosis."""
    args = dict(base_args)
    f1 = prev_results['best_macro_f1']

    for diag_type, diag_msg in diagnoses:
        print(f"  Diagnosis: {diag_type} — {diag_msg}")

    primary = diagnoses[0][0] if diagnoses else 'unknown'

    if primary == 'catastrophic':
        # Start over with very different settings
        args['lr'] = '1e-3'
        args['sigma'] = '3.0'
        args['loss_pos_weight'] = '100'
        args['epochs'] = '100'
        args['extra'] = ''  # strip mantis/jitter, go simple

    elif primary == 'fp_flood':
        # Too many false positives — reduce pos_weight so model is less eager to predict positives
        # (pos_weight amplifies positive pixel loss; lower can reduce FP flood).
        pw = float(args.get('loss_pos_weight', 300))
        args['loss_pos_weight'] = str(max(pw * 0.5, 30))
        sigma = float(args.get('sigma', 2.0))
        args['sigma'] = str(max(sigma - 0.5, 1.0))
        # Lower learning rate for fine-tuning
        args['lr'] = str(float(args.get('lr', 3e-4)) * 0.3)
        args['epochs'] = '100'

    elif primary == 'low_recall':
        # Missing particles — increase sigma for bigger targets, increase pos_weight
        pw = float(args.get('loss_pos_weight', 300))
        args['loss_pos_weight'] = str(min(pw * 2, 1000))
        sigma = float(args.get('sigma', 2.0))
        args['sigma'] = str(min(sigma + 1.0, 5.0))
        args['epochs'] = '150'

    elif primary == '12nm_weak':
        # 12nm particles missed — increase pos_weight aggressively
        args['loss_pos_weight'] = '800'
        sigma = float(args.get('sigma', 2.0))
        args['sigma'] = str(min(sigma + 0.5, 4.0))
        args['lr'] = str(float(args.get('lr', 3e-4)) * 0.5)
        args['epochs'] = '100'

    elif primary == 'both_weak':
        # Nothing works well — try bigger model, more training
        args['base_channels'] = '64'
        args['loss_pos_weight'] = '500'
        args['sigma'] = '2.5'
        args['epochs'] = '200'

    elif primary == 'finetune':
        # Already decent — small LR, more epochs, resume from best
        args['lr'] = str(float(args.get('lr', 3e-4)) * 0.1)
        args['epochs'] = '100'
        args['loss_pos_weight'] = str(float(args.get('loss_pos_weight', 300)))
        # Tighten sigma slightly for better localization
        sigma = float(args.get('sigma', 2.0))
        args['sigma'] = str(max(sigma - 0.3, 1.0))

    elif primary == 'class_imbalance':
        args['loss_pos_weight'] = '600'
        args['lr'] = str(float(args.get('lr', 3e-4)) * 0.3)
        args['epochs'] = '100'

    else:
        # Unknown — try lower LR and more epochs
        args['lr'] = str(float(args.get('lr', 3e-4)) * 0.5)
        args['epochs'] = '150'

    return args


def run_command(cmd, label=""):
    """Run a shell command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  CMD: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode


def main():
    p = argparse.ArgumentParser(description="Auto-optimize: iteratively improve F1")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--model_type", type=str, default="unet_deep")
    p.add_argument("--loss_type", type=str, default="focal_bce")
    p.add_argument("--sigma", type=str, default="2.0")
    p.add_argument("--loss_pos_weight", type=str, default="300")
    p.add_argument("--lr", type=str, default="3e-4")
    p.add_argument("--epochs", type=str, default="200")
    p.add_argument("--patch_h", type=str, default="256")
    p.add_argument("--patch_stride", type=str, default="128")
    p.add_argument("--base_channels", type=str, default="32")
    p.add_argument("--extra_flags", type=str, default="")
    p.add_argument("--target_f1", type=float, default=TARGET_F1)
    p.add_argument("--max_rounds", type=int, default=MAX_ROUNDS)
    args = p.parse_args()

    base_save = args.save_dir
    eval_path = os.path.join(base_save, "eval_results.txt")

    if not os.path.exists(eval_path):
        print(f"ERROR: No eval results at {eval_path}")
        sys.exit(1)

    base_args = {
        'model_type': args.model_type,
        'loss_type': args.loss_type,
        'sigma': args.sigma,
        'loss_pos_weight': args.loss_pos_weight,
        'lr': args.lr,
        'epochs': args.epochs,
        'patch_h': args.patch_h,
        'patch_stride': args.patch_stride,
        'base_channels': args.base_channels,
        'extra': args.extra_flags,
    }

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  AUTO-OPTIMIZE ROUND {round_num}/{args.max_rounds}")
        print(f"{'#'*60}")

        # Parse previous results
        results = parse_eval_results(eval_path)
        f1 = results['best_macro_f1']
        print(f"\n  Previous best macro F1: {f1:.4f} (target: {args.target_f1})")

        if f1 >= args.target_f1:
            print(f"\n  TARGET REACHED! F1={f1:.4f} >= {args.target_f1}")
            break

        # Diagnose and adjust
        diagnoses = diagnose(results)
        retry_args = generate_retry_args(base_args, diagnoses, round_num, results)

        round_dir = os.path.join(base_save, f"round{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        os.makedirs(os.path.join(round_dir, "vis"), exist_ok=True)

        # Save diagnosis
        diag_path = os.path.join(round_dir, "diagnosis.json")
        with open(diag_path, 'w') as f:
            json.dump({
                'round': round_num,
                'prev_f1': f1,
                'diagnoses': diagnoses,
                'adjusted_args': retry_args,
            }, f, indent=2)

        # Resume from best checkpoint
        resume_ckpt = os.path.join(base_save, "detector_best.pt")
        if round_num > 1:
            prev_round = os.path.join(base_save, f"round{round_num-1}")
            if os.path.exists(os.path.join(prev_round, "detector_best.pt")):
                resume_ckpt = os.path.join(prev_round, "detector_best.pt")

        ph = retry_args['patch_h']
        extra = retry_args.get('extra', '')

        # TRAIN
        train_cmd = (
            f"python -u train_detector.py"
            f" --data_root \"{args.data_root}\""
            f" --model_type {retry_args['model_type']}"
            f" --base_channels {retry_args['base_channels']}"
            f" --epochs {retry_args['epochs']}"
            f" --batch_size 8"
            f" --lr {retry_args['lr']}"
            f" --patch_h {ph} --patch_w {ph}"
            f" --train_samples_per_epoch 2048"
            f" --val_samples_per_epoch 256"
            f" --sigma {retry_args['sigma']}"
            f" --loss_type {retry_args['loss_type']}"
            f" --loss_pos_weight {retry_args['loss_pos_weight']}"
            f" --focal_gamma 2.0"
            f" --weight_decay 1e-4"
            f" --grad_clip 1.0"
            f" --sched cosine"
            f" --warmup_epochs 5"
            f" --early_stop_patience 20"
            f" --early_stop_delta 1e-6"
            f" --seed {42 + round_num}"
            f" --save_dir {round_dir}"
            f" --use_sliding_window"
            f" --patch_stride {retry_args['patch_stride']}"
            f" --resume {resume_ckpt}"
            f" {extra}"
        )
        rc = run_command(train_cmd, f"ROUND {round_num} TRAINING")
        if rc != 0:
            print(f"  Training failed with exit code {rc}")
            continue

        # INFER
        best_ckpt = os.path.join(round_dir, "detector_best.pt")
        if not os.path.exists(best_ckpt):
            best_ckpt = resume_ckpt

        infer_cmd = (
            f"python -u infer_detector.py"
            f" --data_root \"{args.data_root}\""
            f" --checkpoint {best_ckpt}"
            f" --model_type {retry_args['model_type']}"
            f" --base_channels {retry_args['base_channels']}"
            f" --out_csv {round_dir}/predictions.csv"
            f" --out_vis_dir {round_dir}/vis"
            f" --tile_h {ph} --tile_w {ph}"
            f" --stride_h {retry_args['patch_stride']}"
            f" --stride_w {retry_args['patch_stride']}"
            f" --threshold 0.01"
            f" --min_distance 3"
            f" --save_vis"
        )
        run_command(infer_cmd, f"ROUND {round_num} INFERENCE")

        # EVAL
        eval_cmd = (
            f"python -u evaluate_detector.py"
            f" --data_root \"{args.data_root}\""
            f" --pred_csv {round_dir}/predictions.csv"
            f" --match_dist 10"
            f" --sweep_start 0.01 --sweep_end 0.5 --sweep_steps 50"
        )
        eval_out = os.path.join(round_dir, "eval_results.txt")
        run_command(f"{eval_cmd} | tee {eval_out}", f"ROUND {round_num} EVALUATION")

        # Update eval_path for next round's diagnosis
        eval_path = eval_out

        # Check if we improved
        new_results = parse_eval_results(eval_path)
        new_f1 = new_results['best_macro_f1']
        print(f"\n  Round {round_num} result: F1={new_f1:.4f} (was {f1:.4f}, delta={new_f1-f1:+.4f})")

        if new_f1 >= args.target_f1:
            print(f"\n  TARGET REACHED! F1={new_f1:.4f} >= {args.target_f1}")
            break

        # Update base_args for next round
        base_args = retry_args

    # Final summary
    print(f"\n{'='*60}")
    print(f"  AUTO-OPTIMIZE COMPLETE")
    print(f"  Rounds run: {min(round_num, args.max_rounds)}")
    final = parse_eval_results(eval_path)
    print(f"  Final best macro F1: {final['best_macro_f1']:.4f}")
    print(f"  Best threshold: {final['best_threshold']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
