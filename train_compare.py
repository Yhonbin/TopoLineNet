"""
train_compare.py
================
Unified, low-coupling training driver for baseline comparison experiments.

It reuses EVERYTHING from train_net.py (logger, early-stopping, validate,
snapshot, checkpoint, schedules) and ONLY swaps the model. This guarantees
that the single experimental variable is the network architecture: identical
data, augmentation, loss (harness_topology_loss), optimizer (AdamW), schedule
(CosineAnnealingLR), AMP, and grad clipping — exactly as in train_net.py.

Examples
--------
# train one baseline with the same hyper-params as train_net.py
python train_compare.py --model unet --epochs 150

# train every registered baseline back-to-back (supervised only)
python train_compare.py --model all

# include your full model under the identical pipeline
python train_compare.py --model ours

# turn Mean-Teacher semi-supervision ON for a baseline (extra ablation)
python train_compare.py --model unet --semi

Output (per model) lands in:
    ./exp_compare/<model>/
        best_model.pth          <- picked by val loss, used by evaluator
        last_model.pth
        history.json
        convergence_curve.pdf
        config.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# --- reuse your existing building blocks verbatim ---------------------------
from Datasets import HarnessDataset, UnlabeledHarnessDataset
from loss import harness_topology_loss
from train_net import (
    ExperimentLogger,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    cycle,
    simulate_hard_occlusion,
    get_current_consistency_weight,
    update_ema_variables,
    validate,
    save_snapshot,
)
from compare_models import build_model, MODEL_REGISTRY, list_models
import ablation_models


# ---------------------------------------------------------------------------
def train_one_model(model_name: str, args) -> Path:
    """Train a single registered model end-to-end. Returns its exp dir."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    exp_dir = Path(args.out_root) / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / "last_model.pth"
    with open(exp_dir / "config.json", "w") as f:
        json.dump({**vars(args), "model": model_name}, f, indent=2)

    print(f"\n{'='*70}\n[TRAIN] model = {model_name}\n"
          f"        out   = {exp_dir}\n        device= {device}\n{'='*70}")

    # ---------------- data (identical to train_net.py) ----------------
    ds_lab = HarnessDataset(args.labeled_dir, augment=True)
    ds_val = HarnessDataset(args.val_dir, augment=False)
    assert len(ds_lab) > 0, "labeled set empty"

    drop_lab = len(ds_lab) >= args.bs_lab
    loader_lab = DataLoader(ds_lab, batch_size=args.bs_lab, shuffle=True,
                            drop_last=drop_lab, num_workers=args.workers,
                            persistent_workers=(args.workers > 0), pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.bs_val, shuffle=False,
                            num_workers=max(1, args.workers // 2),
                            persistent_workers=(args.workers > 0), pin_memory=True)

    # Semi-supervision is OFF by default for baselines (it's part of OUR method).
    use_semi = args.semi
    ema_model = None
    iter_unlab = None
    if use_semi:
        ds_unlab = UnlabeledHarnessDataset(args.unlabeled_dir)
        use_semi = len(ds_unlab) > 0
        if use_semi:
            loader_unlab = DataLoader(ds_unlab, batch_size=args.bs_unlab,
                                      shuffle=True, drop_last=True,
                                      num_workers=max(1, args.workers // 2),
                                      persistent_workers=(args.workers > 0),
                                      pin_memory=True)
            iter_unlab = iter(cycle(loader_unlab))

    # ---------------- model (the ONLY thing that varies) ----------------
    model = build_model(model_name, pretrained=args.pretrained).to(device)

    if use_semi:
        ema_model = build_model(model_name, pretrained=False).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    # ---------------- optim / sched / amp (identical) ----------------
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base_lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    iterations_per_epoch = (max(len(loader_lab), len(loader_unlab))
                            if use_semi else len(loader_lab))

    logger = ExperimentLogger(exp_dir)
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    start_epoch, best_val_loss = 1, float("inf")
    if args.resume and ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            ckpt_path, model, ema_model, optimizer, scheduler, scaler,
            early_stopper, device)
        start_epoch += 1
        print(f"[Resume] from epoch {start_epoch}, best={best_val_loss:.4f}")

    iter_lab = iter(cycle(loader_lab))

    # ---------------- training loop (mirrors train_net.py) ----------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if ema_model is not None:
            ema_model.eval()
        total_loss = total_sup = total_unsup = 0.0
        consist_w = get_current_consistency_weight(
            epoch, args.epochs, args.max_consist_w, args.rampup) if use_semi else 0.0

        pbar = tqdm(range(iterations_per_epoch),
                    desc=f"[{model_name}] Ep {epoch}/{args.epochs}", unit="batch")
        for _ in pbar:
            optimizer.zero_grad(set_to_none=True)
            imgs_lab, t_line = next(iter_lab)
            imgs_lab = imgs_lab.to(device, non_blocking=True)
            t_line = t_line.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                p_line = model(imgs_lab)
                if p_line.shape[-2:] != t_line.shape[-2:]:
                    t_line = F.interpolate(t_line, size=p_line.shape[-2:],
                                           mode="bilinear", align_corners=False)
                supervised_loss = harness_topology_loss(p_line.float(), t_line)

                consistency_loss = torch.tensor(0.0, device=device)
                if use_semi:
                    imgs_unlab = next(iter_unlab).to(device, non_blocking=True)
                    with torch.no_grad():
                        pseudo = ema_model(imgs_unlab).detach()
                    inten = consist_w / max(args.max_consist_w, 1e-6)
                    imgs_strong = simulate_hard_occlusion(imgs_unlab, intensity=inten)
                    p_student = model(imgs_strong)
                    if p_student.shape != pseudo.shape:
                        pseudo = F.interpolate(pseudo, size=p_student.shape[-2:],
                                               mode="bilinear", align_corners=False)
                    consistency_loss = F.mse_loss(p_student.float(),
                                                  pseudo.float()) * 10.0

                loss = supervised_loss + consist_w * consistency_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if use_semi:
                update_ema_variables(model, ema_model, alpha=args.ema_alpha)

            total_loss += loss.item()
            total_sup += supervised_loss.item()
            if use_semi:
                total_unsup += consistency_loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "sup": f"{supervised_loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / iterations_per_epoch
        avg_sup = total_sup / iterations_per_epoch
        avg_unsup = (total_unsup / iterations_per_epoch) if use_semi else 0.0

        if epoch % args.val_every == 0:
            val_loss = validate(model, loader_val, device, use_amp=args.amp)
        else:
            val_loss = (logger.history["val_loss"][-1]
                        if logger.history["val_loss"] else float("inf"))

        logger.update(epoch, avg_loss, avg_sup, avg_unsup, val_loss)
        print(f"[{model_name}][Ep {epoch:3d}] train={avg_loss:.4f} "
              f"(sup={avg_sup:.4f}, unsup={avg_unsup:.4f}) | val={val_loss:.4f} "
              f"| lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.vis_every == 0:
            save_snapshot(imgs_lab[0], t_line[0], p_line[0], epoch, exp_dir,
                          prefix=model_name)

        improved = early_stopper.step(val_loss)
        if improved:
            best_val_loss = val_loss
            state = (model.module.state_dict()
                     if hasattr(model, "module") else model.state_dict())
            torch.save(state, exp_dir / "best_model.pth")
            print(f"   ↑ best updated: {best_val_loss:.4f}")

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(ckpt_path, model, ema_model, optimizer, scheduler,
                            scaler, early_stopper, epoch, best_val_loss)

        if early_stopper.should_stop:
            print(f"[EarlyStop] no improvement in {args.patience} epochs.")
            save_checkpoint(ckpt_path, model, ema_model, optimizer, scheduler,
                            scaler, early_stopper, epoch, best_val_loss)
            break

    logger.plot_loss()
    print(f"[Done] {model_name}: best val={best_val_loss:.4f} -> {exp_dir}")
    return exp_dir


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unet",
                   help="registry key, or 'all' to train every baseline, "
                        "or comma list e.g. 'unet,deeplabv3plus'")
    p.add_argument("--list", action="store_true", help="list models and exit")

    # data (identical defaults to train_net.py)
    p.add_argument("--labeled_dir",   default="./data/train")
    p.add_argument("--unlabeled_dir", default="./data/data_unlabeled")
    p.add_argument("--val_dir",       default="./data/val")
    p.add_argument("--out_root",      default="./exp_compare")

    # hyper-params — MUST mirror train_net.py for a fair comparison
    p.add_argument("--bs_lab",   type=int,   default=8)
    p.add_argument("--bs_unlab", type=int,   default=4)
    p.add_argument("--bs_val",   type=int,   default=8)
    p.add_argument("--epochs",   type=int,   default=150)
    p.add_argument("--base_lr",  type=float, default=2e-4)
    p.add_argument("--wd",       type=float, default=5e-4)
    p.add_argument("--max_consist_w", type=float, default=0.5)
    p.add_argument("--rampup",   type=float, default=0.4)
    p.add_argument("--ema_alpha", type=float, default=0.995)

    # engineering
    p.add_argument("--workers",  type=int, default=2)
    p.add_argument("--amp",      action="store_true", default=True)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--vis_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min_delta", type=float, default=1e-4)

    # behaviour flags
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="load ImageNet encoder weights (matches HRNet setting)")
    p.add_argument("--semi", action="store_true", default=False,
                   help="enable Mean-Teacher for this baseline (extra ablation)")
    p.add_argument("--resume", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_models()
        return

    if args.model == "all":
        names = list(MODEL_REGISTRY.keys())
    elif "," in args.model:
        names = [s.strip() for s in args.model.split(",")]
    else:
        names = [args.model]

    for n in names:
        if n not in MODEL_REGISTRY:
            print(f"[skip] unknown model '{n}'")
            continue
        train_one_model(n, args)


if __name__ == "__main__":
    main()
