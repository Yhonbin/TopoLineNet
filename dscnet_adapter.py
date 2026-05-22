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
    """Import the DSConv_pro operator class (not the full network)."""
    candidates = [
        ("S3_DSConv_pro", "DSConv_pro"),
        ("DSConv_pro",    "DSConv_pro"),
        ("S3_DSConv",     "DSConv"),
        ("DSConv",        "DSConv"),
    ]
    last_err = None
    for mod_name, cls_name in candidates:
        try:
            mod = _load_module_compat(mod_name)
            return getattr(mod, cls_name)
        except (ImportError, AttributeError) as e:
            last_err = e
            continue
    raise ImportError(
        f"Could not import DSConv. Check DSCNET_REPO_PATH.\n"
        f"Last error: {last_err}\n"
        f"Searching: {DSCNET_REPO_PATH}"
    )
    
def _load_module_compat(mod_name: str):
    """
    Import a module by name, with automatic source-patching for Python 3.9
    compatibility (rewrites `str | torch.device` -> `Union[str, torch.device]`)
    and device consistency fixes.
    """
    import importlib
    import importlib.util
    import types
    import re

    if DSCNET_REPO_PATH not in sys.path and os.path.isdir(DSCNET_REPO_PATH):
        sys.path.insert(0, DSCNET_REPO_PATH)

    # ---- attempt 1: plain import (Python 3.10+) ----
    try:
        return importlib.import_module(mod_name)
    except TypeError as e:
        if "unsupported operand type(s) for |" not in str(e):
            raise
    except ImportError:
        raise

    # ---- attempt 2: source-patch for Python 3.9 ----
    # Clear any partial registration from the failed attempt 1.
    sys.modules.pop(mod_name, None)

    spec = importlib.util.find_spec(mod_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Cannot locate source for '{mod_name}'")

    with open(spec.origin, "r", encoding="utf-8") as fh:
        src = fh.read()

    # PATCH 1: Python 3.9 union-type annotation compatibility.
    if "from typing import Union" not in src:
        src = "from typing import Union\n" + src
    src = re.sub(r'\bstr\s*\|\s*torch\.device\b',
                'Union[str, torch.device]', src)
    src = re.sub(r'\bstr\s*\|\s*device\b',
                'Union[str, torch.device]', src)

    # PATCH 2: device consistency — derive device from offset tensor.
    src = re.sub(
        r'^(\s*)device\s*=\s*torch\.device\(device\)\s*$',
        r'\1device = offset.device  # patched: derive from input tensor',
        src, flags=re.MULTILINE,
    )

    mod = types.ModuleType(mod_name)
    mod.__file__ = spec.origin
    exec(compile(src, spec.origin, "exec"), mod.__dict__)
    sys.modules[mod_name] = mod
    return mod
    
    

# ---------------------------------------------------------------------------
# DSConv block: parallel x-snake + y-snake + standard conv, then fuse.
# This mirrors how DSCNet uses DSConv inside its encoder/decoder blocks.
# ---------------------------------------------------------------------------

class DSCNetProWrapper(nn.Module):
    """
    Thin wrapper around the authors' original DSCNet_pro that:
      - fixes the device mismatch on every forward pass
      - outputs logits [B,1,H,W] (sigmoid removed; wrapper adds it)
      - resizes output to match input if they differ
 
    The original DSCNet_pro applies sigmoid internally. We strip it here
    so that CenterlineHeatmapWrapper (apply_sigmoid=True) handles it
    uniformly — same as every other baseline.
    """
 
    def __init__(self, n_channels=3, n_classes=1, kernel_size=9,
                 extend_scope=1.0, if_offset=True, device="cuda",
                 number=32, dim=1):
        super().__init__()
 
        # Import the original network class.
        _load_module_compat("S3_DSConv_pro")
        mod = _load_module_compat("S3_DSCNet_pro")
        DSCNet_pro_cls = getattr(mod, "DSCNet_pro")
 
        self.net = DSCNet_pro_cls(
            n_channels=n_channels,
            n_classes=n_classes,
            kernel_size=kernel_size,
            extend_scope=extend_scope,
            if_offset=if_offset,
            device=device,
            number=number,
            dim=dim,
        )
        # Remove the original sigmoid — our wrapper adds it.
        # We'll intercept in forward instead of modifying the original class.
 
    def _sync_device(self, live_device: torch.device):
        """
        Walk all sub-modules and sync any stored `device` attribute to the
        live device of the input tensor. DSConv_pro stores self.device at
        __init__ time ("cuda"), but .to("cuda:0") doesn't update it.
        """
        for m in self.net.modules():
            if hasattr(m, "device") and not isinstance(m.device, torch.device):
                try:
                    m.device = live_device
                except Exception:
                    pass
            elif hasattr(m, "device") and m.device != live_device:
                m.device = live_device
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._sync_device(x.device)
 
        # The original forward returns sigmoid output. We need logits for
        # the wrapper, so we temporarily replace sigmoid with identity.
        original_sigmoid = self.net.sigmoid
        self.net.sigmoid = nn.Identity()
        try:
            out = self.net(x)
        finally:
            self.net.sigmoid = original_sigmoid
 
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        return out  # logits


def build_dscnet_original(device="cuda", number=32, kernel_size=9):
    """Build the authors' original DSCNet_pro network."""
    return DSCNetProWrapper(
        n_channels=3, n_classes=1, kernel_size=kernel_size,
        extend_scope=1.0, if_offset=True, device=device,
        number=number, dim=1,
    )
    
    
class _DSConvBlock(nn.Module):
    """std_conv + dsconv_x + dsconv_y -> concat -> fuse."""
 
    def __init__(self, in_ch, out_ch, kernel_size=9, extend_scope=1.0,
                 if_offset=True, device="cuda"):
        super().__init__()
        DSConv = _import_dsconv()
        self.conv_std = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.dsconv_x = DSConv(in_ch, out_ch, kernel_size, extend_scope,
                               0, if_offset, device)
        self.dsconv_y = DSConv(in_ch, out_ch, kernel_size, extend_scope,
                               1, if_offset, device)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        live_device = x.device
        for dsconv in (self.dsconv_x, self.dsconv_y):
            if hasattr(dsconv, "device") and dsconv.device != live_device:
                dsconv.device = live_device
        a = self.conv_std(x)
        b = self.dsconv_x(x)
        c = self.dsconv_y(x)
        return self.fuse(torch.cat([a, b, c], dim=1))



class DSCNetSeg(nn.Module):
    """Lightweight DSConv-based U-Net with stem downsampling."""
 
    def __init__(self, in_ch=3, base=32, kernel_size=9, extend_scope=1.0,
                 device="cuda"):
        super().__init__()
        assert base % 4 == 0, f"base must be divisible by 4, got {base}"
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
 
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
        )
        self.enc1 = ds_block(c1, c1)
        self.enc2 = ds_block(c1, c2)
        self.enc3 = ds_block(c2, c3)
        self.enc4 = ds_block(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ds_block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ds_block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = std_block(c1 * 2, c1)
        self.out_conv = nn.Conv2d(c1, 1, 1)
 
    def forward(self, x):
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)
        return out
    
    

def build_dscnet_lite(device="cuda", base=32, kernel_size=9):
    """Build the lightweight DSConv U-Net."""
    return DSCNetSeg(in_ch=3, base=base, kernel_size=kernel_size, device=device)
 