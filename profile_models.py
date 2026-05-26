"""
profile_models.py
=================
Quick profiler: compare parameter count, FLOPs, and per-module GPU timing
for every registered model.  Helps pinpoint WHERE time goes in your network.

Usage:
    python profile_models.py                       # profile all
    python profile_models.py --models unet,ours    # only these two
"""

import argparse
import time
import sys

import torch
import torch.nn as nn
import os
from compare_models import build_model, MODEL_REGISTRY
import ablation_models

# ---------------------------------------------------------------------------
# 1. Parameter count
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}


# ---------------------------------------------------------------------------
# 2. GPU timing (warm-up + averaged forward + backward)
# ---------------------------------------------------------------------------

@torch.no_grad()
def time_forward(model, x, n_warmup=5, n_measure=20):
    """Measure forward-only time on GPU."""
    model.eval()
    for _ in range(n_warmup):
        _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        _ = model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_measure * 1000  # ms


def time_forward_backward(model, x, n_warmup=3, n_measure=10):
    """Measure forward + backward time on GPU."""
    model.train()
    target = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3],
                         device=x.device)
    for _ in range(n_warmup):
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_measure * 1000  # ms


# ---------------------------------------------------------------------------
# 3. Per-module breakdown for HarnessHRNetV2 (your model)
# ---------------------------------------------------------------------------

def _unwrap_harness(model):
    """
    build_ours / _build_ablation wrap the core in CenterlineHeatmapWrapper,
    so the harness lives at model.core. We also tolerate a bare harness or a
    DataParallel wrapper.
    """
    try:
        from HRNet import HarnessHRNetV2
    except ImportError:
        return None
 
    m = model
    # unwrap DataParallel
    if hasattr(m, "module"):
        m = m.module
    # unwrap CenterlineHeatmapWrapper
    if hasattr(m, "core"):
        m = m.core
    return m if isinstance(m, HarnessHRNetV2) else None

def profile_harness_breakdown(model, device, input_size=(1, 3, 512, 512),
                              tag=""):
    """
    Instrument each sub-module of a (possibly ablated) HarnessHRNetV2:
        backbone / CBAM / StripPooling / resize+concat / neck / head.
 
    Modules turned off in an ablation are nn.Identity (CBAM/StripPool) or a
    SimpleNeck (ASPP), so their timing rows will naturally drop toward ~0 or
    reflect the lighter neck — exactly the signal you want for the ablation
    table. The neck row is labelled with the actual neck class name.
    """
    harness = _unwrap_harness(model)
    if harness is None:
        print(f"[skip] {tag or 'model'}: not a HarnessHRNetV2 variant, "
              f"no per-module breakdown.")
        return
 
    harness = harness.to(device).eval()
    x = torch.randn(*input_size, device=device)
    n_warmup, n_measure = 3, 10
 
    # warm up everything
    for _ in range(n_warmup):
        _ = harness(x)
    torch.cuda.synchronize()
 
    # ---- backbone ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        features = harness.backbone(x)
        torch.cuda.synchronize()
    t_backbone = (time.perf_counter() - t0) / n_measure * 1000
 
    # ---- CBAM + StripPooling (per scale; Identity when ablated -> ~0 ms) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        enhanced = []
        for i, f in enumerate(features):
            f_att = harness.attentions[i](f)
            f_strip = harness.strip_pools[i](f_att)
            enhanced.append(f_strip)
        torch.cuda.synchronize()
    t_attn_strip = (time.perf_counter() - t0) / n_measure * 1000
 
    # ---- resize + concat ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        target_size = enhanced[0].shape[-2:]
        resized = [nn.functional.interpolate(
            f, size=target_size, mode="bilinear", align_corners=True)
            for f in enhanced]
        combined = torch.cat(resized, dim=1)
        torch.cuda.synchronize()
    t_resize_cat = (time.perf_counter() - t0) / n_measure * 1000
 
    # ---- neck (ASPP or SimpleNeck) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        neck_out = harness.neck(combined)
        torch.cuda.synchronize()
    t_neck = (time.perf_counter() - t0) / n_measure * 1000
 
    # ---- head (ConvTranspose upsample + conv) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        logits = harness.head_line(neck_out)
        out = torch.sigmoid(nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=True))
        torch.cuda.synchronize()
    t_head = (time.perf_counter() - t0) / n_measure * 1000
 
    total = t_backbone + t_attn_strip + t_resize_cat + t_neck + t_head
 
    # auto-label the neck row by the actual class that was built
    neck_name = type(harness.neck).__name__  # "ASPP" or "SimpleNeck"
    cfg = (f"cbam={harness.use_cbam} strip={harness.use_strip_pool} "
           f"aspp={harness.use_aspp}")
 
    print("\n" + "=" * 64)
    print(f"HarnessHRNetV2 per-module GPU breakdown   {tag}")
    print(f"Config: {cfg}")
    print(f"Input:  {input_size}")
    print("=" * 64)
    for name, t in [
        ("HRNet backbone",      t_backbone),
        ("CBAM + StripPooling", t_attn_strip),
        ("Resize + Concat",     t_resize_cat),
        (f"Neck ({neck_name})", t_neck),
        ("Decode head",         t_head),
        ("TOTAL",               total),
    ]:
        pct = t / total * 100 if total > 0 else 0
        print(f"  {name:22s}  {t:7.2f} ms  ({pct:5.1f}%)")
    print()
def profile_ours_breakdown(device, input_size=(1, 3, 512, 512)):
    """
    Instrument each sub-module of HarnessHRNetV2 individually to see which
    one eats the most time:  backbone / CBAM / StripPooling / ASPP / head.
    """
    model = build_model("ours", pretrained=False)
    profile_harness_breakdown(model, device, input_size=input_size, tag="(ours-full)")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--bs", type=int, default=4, help="batch size for timing")
    ap.add_argument("--size", type=int, default=512, help="input H=W")
    ap.add_argument("--breakdown", action="store_true",
                    help="also print per-module breakdown for every HarnessHRNetV2 "
                         "variant in the selected set (not just the full model)")
    ap.add_argument("--list", action="store_true",
                    help="list all registry keys (incl. ablations) and exit")
    args = ap.parse_args()
    if args.list:
        print(f"{'key':18s} | description")
        print("-" * 72)
        for k, (_, _, desc) in MODEL_REGISTRY.items():
            print(f"{k:18s} | {desc}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] No GPU detected. Timing results will not reflect "
              "real training speed.")

    names = (list(MODEL_REGISTRY.keys()) if args.models == "all"
             else [s.strip() for s in args.models.split(",")])

    x = torch.randn(args.bs, 3, args.size, args.size, device=device)

    print(f"\n{'Model':18s} | {'Params(M)':>10s} | {'Fwd(ms)':>8s} | "
          f"{'Fwd+Bwd(ms)':>12s} | {'vs UNet':>8s}")
    print("-" * 75)

    results = {}
    built_models = {}  # keep references for optional breakdown pass
    for name in names:
        if name not in MODEL_REGISTRY:
            print(f"  {name}: unknown, skipping")
            continue
        try:
            model = build_model(name, pretrained=False).to(device)
        except Exception as e:
            print(f"  {name}: build failed — {e}")
            continue

        params = count_params(model)
        if device.type == "cuda":
            t_fwd = time_forward(model, x)
            t_fb = time_forward_backward(model, x)
        else:
            t_fwd = t_fb = float("nan")
        results[name] = {"params": params, "fwd_ms": t_fwd, "fb_ms": t_fb}
        # Keep harness variants around for breakdown; free everything else.
        if args.breakdown and _unwrap_harness(model) is not None:
            built_models[name] = model
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
 
    # print summary table
    unet_fb = results.get("unet", {}).get("fb_ms", 1.0)
    for name in names:
        if name not in results:
            continue
        r = results[name]
        ratio = (r["fb_ms"] / unet_fb) if (unet_fb and unet_fb > 0) else float("nan")
        print(f"  {name:18s} | {r['params']['trainable_M']:9.2f}M | "
              f"{r['fwd_ms']:7.1f} | {r['fb_ms']:11.1f} | "
              f"{ratio:7.2f}x")

    # per-module breakdown for your model
    if device.type == "cuda":
        if args.breakdown:
            # breakdown for EVERY harness variant in the selected set
            for name, model in built_models.items():
                profile_harness_breakdown(
                    model, device,
                    input_size=(args.bs, 3, args.size, args.size),
                    tag=f"({name})")
                del model
                torch.cuda.empty_cache()
        else:
            # default: just the full 'ours' model, as before
            profile_ours_breakdown(
                device, input_size=(args.bs, 3, args.size, args.size))


if __name__ == "__main__":
    main()
