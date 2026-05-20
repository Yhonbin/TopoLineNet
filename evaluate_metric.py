"""
evaluate_metric.py
==================
Drop-in evaluator for TopoLineNet (and any centerline-extraction baseline).

Usage (single experiment):
    python evaluate_metric.py \
        --model_path ./exp_xxx/best_model.pth \
        --test_dir   ./data/val \
        --method_name "Ours-Full" \
        --out_dir    ./eval_results

Usage (programmatic — for batch comparison / ablation):
    from evaluate_metric import Evaluator
    ev = Evaluator(test_dir="./data/val", out_dir="./eval_results")
    ev.run(model_path="./exp_A/best_model.pth", method_name="HRNet-baseline")
    ev.run(model_path="./exp_B/best_model.pth", method_name="HRNet+clDice")
    ev.run(model_path="./exp_C/best_model.pth", method_name="Ours-Full")
    ev.export_summary()    # writes summary.csv + summary.tex

Outputs (per method):
    eval_results/
        per_image/<method>.csv         — every metric for every image
        summary.csv                    — one row per method, mean ± std
        summary.tex                    — LaTeX-ready table fragment
        meta/<method>.json             — config, runtime, image count
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.morphology import skeletonize

from topo_metrics import compute_all, HEADLINE_METRICS


# ---------------------------------------------------------------------------
# Model-agnostic adapter: anything that maps tensor(B,3,H,W) -> tensor(B,1,H,W)
# in [0,1] is acceptable. This lets you drop in U-Net / DeepLab / SegFormer
# for baselines without touching the evaluator.
# ---------------------------------------------------------------------------

def default_predictor(model, imgs, device):
    """Standard forward; override for models with multiple outputs."""
    return model(imgs.to(device))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Reusable evaluator. Instantiate once with a test set, then call run()
    for every method you want to compare. Results accumulate in self.summary.
    """

    def __init__(self,
                 test_dir: str,
                 out_dir: str = "./eval_results",
                 device: str | None = None,
                 batch_size: int = 1,
                 threshold: float = 0.5,
                 num_workers: int = 2):
        self.test_dir = test_dir
        self.out_dir = Path(out_dir)
        (self.out_dir / "per_image").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "meta").mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.threshold = threshold
        self.num_workers = num_workers

        # Build dataset once
        from Datasets import HarnessDataset
        self.dataset = HarnessDataset(test_dir, augment=False)
        self.loader = DataLoader(self.dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
        print(f"[Evaluator] test set = {test_dir}  ({len(self.dataset)} images)")

        # Accumulator across methods
        self.summary_rows = []     # list of dicts

    # -----------------------------------------------------------------------
    def run(self,
            model,
            method_name: str,
            predictor_fn=default_predictor,
            extra_meta: dict | None = None) -> pd.DataFrame:
        """
        Run a single method end-to-end.

        Parameters
        ----------
        model        : nn.Module already on self.device, .eval()
        method_name  : string, used as identifier in tables / filenames
        predictor_fn : callable(model, imgs, device) -> tensor(B,1,H,W) in [0,1]
        extra_meta   : optional dict logged into meta/<method>.json

        Returns
        -------
        per_image_df : DataFrame, one row per image
        """
        model = model.to(self.device).eval()

        rows = []
        t0 = time.time()
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(tqdm(self.loader,
                                                       desc=f"[eval] {method_name}")):
                preds = predictor_fn(model, imgs, self.device)
                # Handle (logits vs sigmoid) — assume already in [0,1]
                for b in range(preds.shape[0]):
                    pred_np = preds[b, 0].cpu().numpy()
                    gt_np = targets[b, 0].cpu().numpy()

                    # Skeletonize both sides identically; this is the only
                    # place post-processing happens, so all methods are
                    # compared fairly.
                    pred_skel = skeletonize(pred_np > self.threshold).astype(np.uint8)
                    gt_skel = skeletonize(gt_np > 0.5).astype(np.uint8)

                    if gt_skel.sum() == 0:
                        continue  # ignore degenerate GT

                    m = compute_all(pred_skel, gt_skel)
                    m["image_idx"] = idx * self.batch_size + b
                    rows.append(m)

        elapsed = time.time() - t0
        per_image_df = pd.DataFrame(rows)
        per_image_path = self.out_dir / "per_image" / f"{method_name}.csv"
        per_image_df.to_csv(per_image_path, index=False)
        print(f"[Evaluator] {method_name}: per-image -> {per_image_path}")

        # ----- aggregate -----
        agg = {"method": method_name}
        for col in per_image_df.columns:
            if col in ("image_idx",):
                continue
            vals = per_image_df[col].values.astype(float)
            agg[f"{col}_mean"] = float(np.mean(vals))
            agg[f"{col}_std"] = float(np.std(vals))
        agg["n_images"] = int(len(per_image_df))
        agg["elapsed_sec"] = float(elapsed)
        agg["ms_per_image"] = float(elapsed / max(1, len(per_image_df)) * 1000)
        self.summary_rows.append(agg)

        # ----- meta -----
        meta = {
            "method": method_name,
            "test_dir": str(self.test_dir),
            "device": str(self.device),
            "threshold": self.threshold,
            "n_images": agg["n_images"],
            "elapsed_sec": agg["elapsed_sec"],
            "ms_per_image": agg["ms_per_image"],
            "params_M": _count_params(model),
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(self.out_dir / "meta" / f"{method_name}.json", "w") as f:
            json.dump(meta, f, indent=2)

        # ----- pretty print -----
        self._print_method_summary(method_name, agg)
        return per_image_df

    # -----------------------------------------------------------------------
    def export_summary(self,
                       csv_name: str = "summary.csv",
                       tex_name: str = "summary.tex",
                       headline_only: bool = True):
        """
        Write a cross-method summary table.
        - csv  : machine-readable, every (mean, std) column.
        - tex  : LaTeX fragment with headline metrics only, formatted "mean±std".
        """
        if not self.summary_rows:
            print("[Evaluator] nothing to export.")
            return None

        df = pd.DataFrame(self.summary_rows)
        df.to_csv(self.out_dir / csv_name, index=False)
        print(f"[Evaluator] summary -> {self.out_dir / csv_name}")

        # LaTeX fragment with the headline metrics
        cols = HEADLINE_METRICS if headline_only else \
            [c[:-5] for c in df.columns if c.endswith("_mean")]
        lines = []
        lines.append(r"\begin{tabular}{l" + "c" * len(cols) + "}")
        lines.append(r"\toprule")
        lines.append("Method & " + " & ".join(cols) + r" \\")
        lines.append(r"\midrule")
        for _, row in df.iterrows():
            cells = [row["method"]]
            for c in cols:
                m = row.get(f"{c}_mean", float("nan"))
                s = row.get(f"{c}_std", float("nan"))
                cells.append(f"{m:.3f}$\\pm${s:.3f}")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        with open(self.out_dir / tex_name, "w") as f:
            f.write("\n".join(lines))
        print(f"[Evaluator] LaTeX  -> {self.out_dir / tex_name}")
        return df

    # -----------------------------------------------------------------------
    def _print_method_summary(self, name, agg):
        print(f"\n=== {name} ===")
        for k in HEADLINE_METRICS:
            mk, sk = f"{k}_mean", f"{k}_std"
            if mk in agg:
                print(f"  {k:14s} : {agg[mk]:.4f} ± {agg[sk]:.4f}")
        print(f"  inference     : {agg['ms_per_image']:.1f} ms/img")
        print("")


# ---------------------------------------------------------------------------
def _count_params(model) -> float:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return float(n / 1e6)


# ---------------------------------------------------------------------------
# CLI for single-method evaluation (backwards-compatible with old usage)
# ---------------------------------------------------------------------------

def _build_default_model(model_path, device):
    """Default loader for HarnessHRNetV2; override for other architectures."""
    from HRNet import HarnessHRNetV2
    model = HarnessHRNetV2(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device)
    # tolerate both DataParallel and bare state_dicts
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./exp_20260519_112009/best_model.pth")
    parser.add_argument("--test_dir", default="./data/val")
    parser.add_argument("--method_name", default="model")
    parser.add_argument("--out_dir", default="./eval_results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ev = Evaluator(test_dir=args.test_dir,
                   out_dir=args.out_dir,
                   device=args.device,
                   threshold=args.threshold)
    model = _build_default_model(args.model_path, ev.device)
    ev.run(model, method_name=args.method_name,
           extra_meta={"checkpoint": args.model_path})
    ev.export_summary()


if __name__ == "__main__":
    main()