"""
dscnet_adapter.py
=================
Self-contained adapter that turns the official Dynamic Snake Convolution
(DSConv, Qi et al. ICCV 2023) operator into a centerline-segmentation network
obeying our unified contract:

    forward(x: [B,3,H,W]) -> [B,1,H,W] logits  (sigmoid added by wrapper)

DESIGN — why a separate file?
-----------------------------
DSCNet's official repo is a tightly-coupled "edit-the-todo-and-run" script
collection. We do NOT import that training machinery. We import ONLY the
DSConv operator (the actual contribution), and build a clean DSCNet-style
U-Net around it ourselves. This keeps coupling to a SINGLE point:
`_import_dsconv()` below. Everything DSCNet-specific lives in this file, so
compare_models.py stays clean and the other baselines are unaffected whether
or not DSCNet is installed.

SETUP
-----
1. git clone https://github.com/YaoleiQi/DSCNet
2. Locate the DSConv operator file, e.g.:
       DSCNet/DSCNet_2D_opensource/Code/DRIVE/S3_DSConv.py
3. Either:
   (a) add that folder to PYTHONPATH, or
   (b) copy the DSConv .py file next to this adapter, or
   (c) edit DSCNET_REPO_PATH below to point at the folder.

The ONLY thing you may need to adjust is `_import_dsconv()` — the class name
and constructor signature differ between DSCNet versions. Adjust there once;
nothing else changes.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# THE SINGLE COUPLING POINT.
# Point this at your cloned repo folder that contains the DSConv operator.
# ---------------------------------------------------------------------------
DSCNET_REPO_PATH = os.path.expanduser(
    "./DSCNet/DSCNet_2D_opensource/Code/DRIVE"
)


def _import_dsconv():
    """
    Import the DSConv operator class from the cloned DSCNet repo.

    Returns a callable/class `DSConv` with (most commonly) this signature in
    the optimised 2D release:

        DSConv(in_ch, out_ch, kernel_size, extend_scope, morph,
               if_offset, device)

    where `morph` = 0 (x-axis snake) or 1 (y-axis snake).

    IF YOUR VERSION DIFFERS: this is the one place to fix. Read your repo's
    DSConv file, then adjust the import name and, if needed, the wrapper in
    `_DSConvBlock` below.
    """
    if DSCNET_REPO_PATH not in sys.path and os.path.isdir(DSCNET_REPO_PATH):
        sys.path.insert(0, DSCNET_REPO_PATH)

    # Try the known module/class names across DSCNet versions, priority order.
    # PRO version first: it uses einops + grid_sample (fast). Requires `einops`.
    candidates = [
        ("S3_DSConv_pro", "DSConv_pro"),  # optimised 2D release (RECOMMENDED)
        ("DSConv_pro", "DSConv_pro"),     # copied next to adapter
        ("S3_DSConv", "DSConv"),          # legacy hand-written interpolation
        ("DSConv", "DSConv"),
    ]
    
    def _load_with_compat(mod_name, cls_name):
        """
        Try normal import first. If it fails with a TypeError caused by
        Python 3.9's inability to parse `X | Y` type unions at class
        definition time, reload the module from source with the union
        syntax rewritten to typing.Union so it parses correctly.
        """
        import importlib
        import importlib.util
        import types
 
        # ---- attempt 1: plain import (works on Python 3.10+) ----
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)
        except TypeError as e:
            if "unsupported operand type(s) for |" not in str(e):
                raise
            # Fall through to source-patching path.
        except (ImportError, AttributeError):
            raise  # real missing-module error, propagate
 
        # ---- attempt 2: locate source file and patch in memory ----
        spec = importlib.util.find_spec(mod_name)
        if spec is None or spec.origin is None:
            raise ImportError(f"Cannot locate source for '{mod_name}'")
 
        with open(spec.origin, "r", encoding="utf-8") as fh:
            src = fh.read()
 
        # Replace every bare `str | torch.device` and `str | device` union
        # with a typing.Union form that Python 3.9 can parse.
        import re
        # Insert `from typing import Union` after the last existing import if
        # not already present.
        if "from typing import Union" not in src and "Union" not in src:
            src = "from typing import Union\n" + src
 
        # Pattern: replace  `str | torch.device`  ->  `Union[str, torch.device]`
        #          replace  `str | device`         ->  `Union[str, device]`
        src = re.sub(
            r'\bstr\s*\|\s*torch\.device\b',
            'Union[str, torch.device]',
            src,
        )
        src = re.sub(
            r'\bstr\s*\|\s*device\b',
            'Union[str, torch.device]',
            src,
        )
 
        # Compile and exec into a fresh module namespace.
        mod = types.ModuleType(mod_name)
        mod.__file__ = spec.origin
        # Make sure the module can find its own imports (torch, einops, etc.)
        exec(compile(src, spec.origin, "exec"), mod.__dict__)  # noqa: S102
        # Register so subsequent imports resolve correctly.
        sys.modules[mod_name] = mod
        return getattr(mod, cls_name)
    
    
    last_err = None
    for mod_name, cls_name in candidates:
        try:
            return _load_with_compat(mod_name, cls_name)
        except (ImportError, AttributeError) as e:
            last_err = e
            continue

    raise ImportError(
        "Could not import DSConv. Set DSCNET_REPO_PATH in dscnet_adapter.py "
        f"to the folder containing the DSConv operator. Last error: {last_err}\n"
        f"Currently looking in: {DSCNET_REPO_PATH}"
    )


# ---------------------------------------------------------------------------
# DSConv block: parallel x-snake + y-snake + standard conv, then fuse.
# This mirrors how DSCNet uses DSConv inside its encoder/decoder blocks.
# ---------------------------------------------------------------------------

class _DSConvBlock(nn.Module):
    """
    One DSCNet-style feature block:
        - a standard 3x3 conv branch
        - a DSConv branch along x  (morph=0)
        - a DSConv branch along y  (morph=1)
      concatenated and fused by 1x1 conv -> BN -> ReLU.

    If your DSConv signature differs, adjust ONLY the two DSConv(...) calls.
    """

    def __init__(self, in_ch, out_ch, kernel_size=9, extend_scope=1.0,
                 if_offset=True, device="cuda"):
        super().__init__()
        DSConv = _import_dsconv()

        self.conv_std = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # morph=0: snake along x-axis; morph=1: snake along y-axis
        # NOTE: signature below matches the optimised 2D release. Adjust if needed.
        self.dsconv_x = DSConv(in_ch, out_ch, kernel_size, extend_scope,
                               0, if_offset, device)
        self.dsconv_y = DSConv(in_ch, out_ch, kernel_size, extend_scope,
                               1, if_offset, device)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        live_device = x.device
        for dsconv in (self.dsconv_x, self.dsconv_y):
            if hasattr(dsconv, "device") and dsconv.device != live_device:
                dsconv.device = live_device
        a = self.conv_std(x)
        b = self.dsconv_x(x)
        c = self.dsconv_y(x)
        return self.fuse(torch.cat([a, b, c], dim=1))


# ---------------------------------------------------------------------------
# A compact DSCNet-style encoder-decoder U-Net using DSConv blocks.
# Deliberately mid-size so it is comparable to U-Net/our model in param count.
# ---------------------------------------------------------------------------

class DSCNetSeg(nn.Module):
    """
    DSConv-based U-Net for single-channel centerline segmentation.
    Output: logits [B,1,H,W] (wrapper adds sigmoid).
    """

    def __init__(self, in_ch=3, base=32, kernel_size=9, extend_scope=1.0,
                 device="cuda"):
        super().__init__()
        # DSConv (both legacy & pro) build GroupNorm(out_ch // 4, out_ch),
        # so every channel count below MUST be divisible by 4.
        assert base % 4 == 0, f"base must be divisible by 4, got {base}"
        self.device = device
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        def ds_block(i, o):
            return _DSConvBlock(i, o, kernel_size, extend_scope, True, device)

        def std_block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            )

        # Stem: cheap standard conv that downsamples 1->1/4 BEFORE any DSConv.
        # DSConv is expensive (offset learning + grid_sample); running it at
        # full 512x512 OOMs. DSCNet-style nets always apply snake conv at
        # reduced resolution. This keeps the comparison fair AND runnable.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=2, padding=1, bias=False),  # 1/2
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),     # 1/4
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
        )

        # Encoder — DSConv blocks operate from 1/4 resolution downward.
        self.enc1 = ds_block(c1, c1)
        self.enc2 = ds_block(c1, c2)
        self.enc3 = ds_block(c2, c3)
        self.enc4 = ds_block(c3, c4)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ds_block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ds_block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = std_block(c1 * 2, c1)   # last block cheap std conv

        self.out_conv = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        s = self.stem(x)            # 1/4 resolution

        e1 = self.enc1(s)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)
        # d1 is at 1/4 resolution -> upsample back to input size.
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        return out  # logits


def build_dscnet_core(device="cuda", base=32, kernel_size=9):
    """Factory used by compare_models.build_dscnet."""
    return DSCNetSeg(in_ch=3, base=base, kernel_size=kernel_size, device=device)
