"""
compare_models.py
=================
Model zoo / factory for baseline comparison experiments.

Design goal
-----------
EVERY model exposed by this module obeys the SAME contract as
HarnessHRNetV2 in HRNet.py:

    forward(x: Tensor[B, 3, H, W]) -> Tensor[B, 1, H, W],  values in [0, 1]

i.e. single-channel centerline heatmap, already passed through sigmoid,
spatially aligned to the input resolution.

Because of this, NONE of the existing code needs to change:
  - train_net.py's loss functions (harness_topology_loss) work as-is
  - evaluate_metric.py's Evaluator works as-is (it assumes outputs in [0,1])

Adding a new baseline = adding one entry to MODEL_REGISTRY. That is the
only coupling point.

Dependencies
------------
    pip install segmentation-models-pytorch timm

Offline note
------------
smp downloads ImageNet encoder weights on first build into
    ~/.cache/torch/hub/checkpoints/
If your training machine has no internet, build each model once on a
machine that does, then copy that cache folder over.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adapter: force any segmentation backbone into the unified contract.
# ---------------------------------------------------------------------------

class CenterlineHeatmapWrapper(nn.Module):
    """
    Wrap an arbitrary segmentation network so that its output is:
        - single channel (B, 1, H, W)
        - sigmoid-activated  -> values in [0, 1]
        - resized to match the input spatial size

    Parameters
    ----------
    core : nn.Module
        Underlying network. Must return either a Tensor or a (Tensor, ...)
        tuple/list whose first element is the main logit map of shape
        (B, C, h, w) with C >= 1.
    apply_sigmoid : bool
        If the core already applies sigmoid internally, set False to avoid
        double activation. Default True (cores are built to return logits).
    """

    def __init__(self, core: nn.Module, apply_sigmoid: bool = True):
        super().__init__()
        self.core = core
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core(x)

        # Some networks (aux heads, deep supervision) return tuples/lists.
        if isinstance(out, (tuple, list)):
            out = out[0]

        # Reduce to a single channel if the core emits multi-channel logits.
        if out.shape[1] > 1:
            out = out[:, :1]

        # Align to input resolution (smp models already match, but be safe).
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:],
                                mode="bilinear", align_corners=False)

        if self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out


# ---------------------------------------------------------------------------
# Builders.  Each returns an nn.Module already wrapped to the contract.
# Keep every builder import-local so that a missing optional dependency only
# breaks the model that needs it, not the whole module.
# ---------------------------------------------------------------------------

def _build_smp(arch: str, encoder: str, pretrained: bool) -> nn.Module:
    """
    Generic builder for any segmentation_models_pytorch architecture.

    Pretrained encoder weight loading strategy (in priority order):
      1. LOCAL_WEIGHTS dict below — explicit local path for this encoder name.
         Supports any .pth downloaded from smp-hub / PyTorch model zoo.
      2. smp built-in download — only reached when no local path is registered
         and pretrained=True. Will attempt network access.
      3. Random init — when pretrained=False.

    To add a new local weight file, append an entry to LOCAL_WEIGHTS:
        "resnet101": "/path/to/resnet101-xxx.pth",
    """
    import segmentation_models_pytorch as smp
    import os

    # ------------------------------------------------------------------ #
    # Edit this dict to register your locally cached encoder weights.
    # Key   = encoder name as passed to smp (e.g. "resnet34", "resnet50")
    # Value = absolute or ~ path to the .pth file
    # ------------------------------------------------------------------ #
    LOCAL_WEIGHTS: dict[str, str] = {
        "resnet34": "~/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth",
        "resnet50": "~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth",
    }

    factory = getattr(smp, arch)

    if pretrained and encoder in LOCAL_WEIGHTS:
        local_path = os.path.expanduser(LOCAL_WEIGHTS[encoder])
        if os.path.isfile(local_path):
            # Step 1: build the network structure WITHOUT downloading anything.
            core = factory(
                encoder_name=encoder,
                encoder_weights=None,   # random init first
                in_channels=3,
                classes=1,
                activation=None,
            )
            # Step 2: load encoder weights from local file.
            # The file is a plain ImageNet classification state_dict, so we
            # only load into the encoder sub-module (decoder stays random-init,
            # which is correct — decoder is always trained from scratch).
            state = torch.load(local_path, map_location="cpu",weights_only=False)
            # Unwrap common wrappers (DataParallel, nested 'model' key, etc.)
            if "state_dict" in state:
                state = state["state_dict"]
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            # 新：兼容两种版本
            result = core.encoder.load_state_dict(state, strict=False)
            if result is not None:
                missing, unexpected = result
                # 打印详细信息
            else:
                # 旧版 PyTorch，只打印简单提示
                print(f"[pretrain] {encoder}: weights loaded from {local_path}")
        else:
            print(f"[pretrain] {encoder}: local path not found ({local_path}), "
                  f"falling back to smp download.")
            core = factory(encoder_name=encoder, encoder_weights="imagenet",
                           in_channels=3, classes=1, activation=None)

    elif pretrained:
        # No local path registered; let smp download (requires network).
        print(f"[pretrain] {encoder}: no local path registered, "
              f"downloading via smp (needs network).")
        core = factory(encoder_name=encoder, encoder_weights="imagenet",
                       in_channels=3, classes=1, activation=None)

    else:
        core = factory(encoder_name=encoder, encoder_weights=None,
                       in_channels=3, classes=1, activation=None)

    return CenterlineHeatmapWrapper(core, apply_sigmoid=True)


def build_unet(pretrained: bool = True, encoder: str = "resnet34") -> nn.Module:
    """Classic U-Net (Ronneberger 2015).  General-purpose baseline."""
    return _build_smp("Unet", encoder, pretrained)


def build_unetpp(pretrained: bool = True, encoder: str = "resnet34") -> nn.Module:
    """U-Net++ (Zhou 2018).  Stronger skip-connection baseline."""
    return _build_smp("UnetPlusPlus", encoder, pretrained)


def build_deeplabv3plus(pretrained: bool = True, encoder: str = "resnet50") -> nn.Module:
    """
    DeepLabV3+ (Chen 2018).  Atrous-conv baseline; conceptually closest to
    our ASPP neck, so a fair "is ASPP alone enough?" reference.
    """
    return _build_smp("DeepLabV3Plus", encoder, pretrained)


def build_linknet(pretrained: bool = True, encoder: str = "resnet34") -> nn.Module:
    """
    LinkNet (Chaurasia 2017).  This is the architectural ancestor of
    D-LinkNet, the classic road-extraction model -> sensible curvilinear
    baseline without needing the custom D-LinkNet repo.
    """
    return _build_smp("Linknet", encoder, pretrained)


def build_hrnet_vanilla(pretrained: bool = True) -> nn.Module:
    """
    Plain HRNet-W18 with a minimal segmentation head, NO CBAM / StripPooling /
    ASPP.  This is the most important reference: it isolates the contribution
    of our architectural additions (doubles as the backbone ablation row).
    """
    core = _VanillaHRNetSeg(model_name="hrnet_w18", pretrained=pretrained)
    # core already returns logits at input resolution -> wrap + sigmoid
    return CenterlineHeatmapWrapper(core, apply_sigmoid=True)


class _VanillaHRNetSeg(nn.Module):
    """
    Bare-bones HRNet segmentation: timm backbone -> concat multiscale feats
    -> 1x1 fuse -> upsample to input. Deliberately plain to serve as the
    'no bells and whistles' control for HarnessHRNetV2.
    """

    def __init__(self, model_name: str = "hrnet_w18", pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            model_name, features_only=True)
        if (pretrained):
            self.backbone.load_state_dict(torch.load('./pretrained_model/hrnetv2_w18_imagenet_pretrained.pth'), strict=False)
        chs = self.backbone.feature_info.channels()
        total = sum(chs)
        self.head = nn.Sequential(
            nn.Conv2d(total, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        target = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=target, mode="bilinear",
                               align_corners=False) for f in feats]
        fused = torch.cat(feats, dim=1)
        out = self.head(fused)
        out = F.interpolate(out, size=x.shape[-2:],
                            mode="bilinear", align_corners=False)
        return out   # logits; wrapper applies sigmoid


def build_dscnet(pretrained: bool = False, number: int = 32,
                 kernel_size: int = 9) -> nn.Module:
    """
    DSCNet — ORIGINAL architecture from the authors' S3_DSCNet_pro.py
    (Qi et al. ICCV 2023). This is the published network, not a reconstruction.
    Recommended for the main comparison table.
 
    pretrained is ignored: DSConv has no ImageNet weights.
    `number` controls the base channel width (authors' default = 32).
    """
    from dscnet_adapter import build_dscnet_original
    device = "cuda" if torch.cuda.is_available() else "cpu"
    core = build_dscnet_original(device=device, number=number,
                                 kernel_size=kernel_size)
    # DSCNetProWrapper returns logits (sigmoid stripped) -> wrapper adds sigmoid.
    return CenterlineHeatmapWrapper(core, apply_sigmoid=True)

def build_dscnet_lite(pretrained: bool = False, base: int = 32,
                      kernel_size: int = 9) -> nn.Module:
    """
    Lightweight DSConv-based U-Net (our construction, NOT the original paper).
    Only uses the DSConv_pro operator; architecture is ours. Smaller, faster,
    useful for ablation on the operator itself vs full DSCNet architecture.
    """
    from dscnet_adapter import build_dscnet_lite as _build_lite
    device = "cuda" if torch.cuda.is_available() else "cpu"
    core = _build_lite(device=device, base=base, kernel_size=kernel_size)
    return CenterlineHeatmapWrapper(core, apply_sigmoid=True)

def build_ours(pretrained: bool = True) -> nn.Module:
    """
    Our full model.  Imported from your existing HRNet.py untouched.
    Note: HarnessHRNetV2 ALREADY applies sigmoid internally, so we tell the
    wrapper NOT to apply it again. The wrapper still normalises the output
    shape, keeping the contract identical to every baseline.
    """
    from HRNet import HarnessHRNetV2
    core = HarnessHRNetV2(pretrained=pretrained)
    return CenterlineHeatmapWrapper(core, apply_sigmoid=False)


# ---------------------------------------------------------------------------
# Registry — the single place you edit to add a baseline.
# value = (builder_fn, kwargs, short_description_for_paper)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "unet":          (build_unet,          {"encoder": "resnet34"},
                      "U-Net (ResNet34), general segmentation baseline"),
    "unetpp":        (build_unetpp,        {"encoder": "resnet34"},
                      "U-Net++ (ResNet34), nested skip connections"),
    "deeplabv3plus": (build_deeplabv3plus, {"encoder": "resnet50"},
                      "DeepLabV3+ (ResNet50), atrous-conv reference for ASPP"),
    "linknet":       (build_linknet,       {"encoder": "resnet34"},
                      "LinkNet (ResNet34), road-extraction style baseline"),
    "hrnet_vanilla": (build_hrnet_vanilla, {},
                      "Plain HRNet-W18, no CBAM/StripPool/ASPP (backbone ablation)"),
    "dscnet":        (build_dscnet,        {"number": 32, "kernel_size": 9},
                      "DSCNet (Original, Qi et al. ICCV2023), tubular SOTA"),
    "dscnet_lite":   (build_dscnet_lite,   {"base": 32, "kernel_size": 9},
                      "DSConv U-Net (lite, our construction), operator ablation"),
    "ours":          (build_ours,          {},
                      "TopoLineNet (Ours-Full): HRNet+CBAM+StripPool+ASPP"),
}


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory entry point.

    Parameters
    ----------
    name : str  — key in MODEL_REGISTRY
    pretrained : bool — load ImageNet encoder weights (default True, matches
                        train_net.py's HRNet setting for a fair comparison)
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    builder, kwargs, _desc = MODEL_REGISTRY[name]
    return builder(pretrained=pretrained, **kwargs)


def list_models() -> None:
    """Print the registry as a readable table (for --list)."""
    print(f"{'key':16s} | description")
    print("-" * 70)
    for k, (_, _, desc) in MODEL_REGISTRY.items():
        print(f"{k:16s} | {desc}")


if __name__ == "__main__":
    # Smoke test: build every model on CPU and check the output contract.
    list_models()
    print("\n[smoke test] checking I/O contract for each model ...")
    x = torch.randn(2, 3, 256, 256)
    for name in MODEL_REGISTRY:
        try:
            m = build_model(name, pretrained=False).eval()
            with torch.no_grad():
                y = m(x)
            ok = (y.shape == (2, 1, 256, 256)
                  and float(y.min()) >= 0.0 and float(y.max()) <= 1.0)
            print(f"  {name:16s} -> {tuple(y.shape)}  "
                  f"range[{y.min():.3f},{y.max():.3f}]  {'OK' if ok else 'FAIL'}")
        except Exception as e:
            print(f"  {name:16s} -> ERROR: {e}")
