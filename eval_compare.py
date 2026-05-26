"""
eval_compare.py
===============
Batch evaluator for all comparison experiments.

It reuses evaluate_metric.Evaluator UNCHANGED. For every model directory
produced by train_compare.py it:
    1. rebuilds the architecture via compare_models.build_model
    2. loads best_model.pth
    3. runs the existing Evaluator (clDice / F1 / HD95 / Betti / APLS / Junction)
    4. accumulates rows and writes summary.csv + summary.tex (one table, all
       methods, mean +- std) — paper-ready.

Because every model obeys the same (B,3,H,W)->(B,1,H,W) in [0,1] contract,
the default_predictor inside evaluate_metric works for all of them.

Examples
--------
# evaluate every trained baseline under ./exp_compare
python eval_compare.py --exp_root ./exp_compare --test_dir ./data/val

# evaluate a subset
python eval_compare.py --models unet,deeplabv3plus,ours
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_metric import Evaluator
from compare_models import build_model, MODEL_REGISTRY
import ablation_models

def _load_into(model, ckpt_path, device):
    """Load a state_dict saved by train_compare.py, tolerant to DP prefixes."""
    state = torch.load(ckpt_path, map_location=device)
    # tolerate DataParallel 'module.' prefix; wrapper 'core.' prefix is kept
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"    [warn] missing keys: {len(missing)} (showing 3) {missing[:3]}")
    if unexpected:
        print(f"    [warn] unexpected keys: {len(unexpected)} (showing 3) {unexpected[:3]}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="./exp_compare",
                    help="root dir that contains one subfolder per model")
    ap.add_argument("--test_dir", default="./data/val")
    ap.add_argument("--out_dir",  default="./eval_results_compare")
    ap.add_argument("--models", default="all",
                    help="'all' or comma list of registry keys to evaluate")
    ap.add_argument("--ckpt_name", default="best_model.pth")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    exp_root = Path(args.exp_root)

    if args.models == "all":
        names = list(MODEL_REGISTRY.keys())
    else:
        names = [s.strip() for s in args.models.split(",")]

    # One Evaluator instance -> shared test set, accumulates all methods.
    ev = Evaluator(test_dir=args.test_dir, out_dir=args.out_dir,
                   device=str(device), threshold=args.threshold)

    for name in names:
        ckpt = exp_root / name / args.ckpt_name
        if not ckpt.exists():
            print(f"[skip] {name}: checkpoint not found at {ckpt}")
            continue
        print(f"\n[eval] building + loading: {name}")
        model = build_model(name, pretrained=False).to(device)
        model = _load_into(model, ckpt, device).eval()
        # Evaluator.run uses default_predictor: model(imgs) -> [0,1] heatmap.
        ev.run(model, method_name=name,
               extra_meta={"checkpoint": str(ckpt)})

    ev.export_summary()   # writes summary.csv + summary.tex across all methods
    print(f"\n[Done] comparison table -> {Path(args.out_dir) / 'summary.csv'}")


if __name__ == "__main__":
    main()
