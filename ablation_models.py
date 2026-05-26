"""
ablation_models.py
==================
消融实验插件 (add-on)。

设计目标
--------
  1. 复用 compare_models 里的 CenterlineHeatmapWrapper / build_model / MODEL_REGISTRY
  2. 新增针对 HarnessHRNetV2 三个模块 (CBAM / StripPooling / ASPP) 的消融 builder
  3. 把这些消融条目注册进 MODEL_REGISTRY，使得 train_compare.py / eval_compare.py /
     profile_models.py 全部无需改动即可直接训练 / 评估 / 计时这些消融配置。

只要在任何入口脚本运行前 `import ablation_models`，消融条目就会自动出现在
MODEL_REGISTRY 中。train_compare.py、eval_compare.py、profile_models.py 都已经
`from compare_models import build_model, MODEL_REGISTRY`，所以最简单的接入方式见文末。

依赖前提
--------
HRNet.py 中的 HarnessHRNetV2 已支持以下关键字参数 (即带消融开关的版本):
    use_cbam: bool, use_strip_pool: bool, use_aspp: bool, aspp_dilations: tuple

新增的 registry key
-------------------
    ours_no_cbam      去掉 CBAM
    ours_no_strip     去掉 StripPooling
    ours_no_aspp      去掉 ASPP (用 SimpleNeck 替代, 保证 head 输入通道不变)
    ours_backbone     三个模块全去掉 (= 纯 HRNet backbone + head)
    ours_aspp_small   保留全部模块, 但把 ASPP 空洞率改小为 (2, 4) — 验证
                      "大空洞率对细线不友好" 这一假设
"""

from __future__ import annotations

import torch.nn as nn

# 复用 compare_models 里已有的契约包装器与注册表 (不修改原文件)
from compare_models import CenterlineHeatmapWrapper, MODEL_REGISTRY


# ---------------------------------------------------------------------------
# 消融 builder —— 全部基于带开关的 HarnessHRNetV2
# 注意:HarnessHRNetV2 内部已做 sigmoid, 故 wrapper apply_sigmoid=False
#       与 compare_models.build_ours 保持完全一致。
# ---------------------------------------------------------------------------

def _build_ablation(pretrained: bool = True,
                    use_cbam: bool = True,
                    use_strip_pool: bool = True,
                    use_aspp: bool = True,
                    aspp_dilations: tuple = (6, 12)) -> nn.Module:
    """统一的消融构建入口。所有消融条目都走这里, 只是开关不同。"""
    from HRNet import HarnessHRNetV2
    core = HarnessHRNetV2(
        pretrained=pretrained,
        use_cbam=use_cbam,
        use_strip_pool=use_strip_pool,
        use_aspp=use_aspp,
        aspp_dilations=aspp_dilations,
    )
    return CenterlineHeatmapWrapper(core, apply_sigmoid=False)


# ---------------------------------------------------------------------------
# 把消融条目注册进 MODEL_REGISTRY
# value = (builder_fn, kwargs, description)  —— 与原 registry 完全同构
# 使用 setdefault 以免重复 import 时报错 / 覆盖已有键。
# ---------------------------------------------------------------------------

_ABLATION_ENTRIES = {
    "ours_no_cbam": (
        _build_ablation, {"use_cbam": False},
        "Ours w/o CBAM (StripPool + ASPP retained)"),
    "ours_no_strip": (
        _build_ablation, {"use_strip_pool": False},
        "Ours w/o StripPooling (CBAM + ASPP retained)"),
    "ours_no_aspp": (
        _build_ablation, {"use_aspp": False},
        "Ours w/o ASPP (SimpleNeck instead; CBAM + StripPool retained)"),
    "ours_backbone": (
        _build_ablation,
        {"use_cbam": False, "use_strip_pool": False, "use_aspp": False},
        "Ours backbone-only (all 3 modules off)"),
    "ours_aspp_small": (
        _build_ablation, {"aspp_dilations": (2, 4)},
        "Ours full but ASPP dilations=(2,4) (thin-structure friendly test)"),
}

for _key, _entry in _ABLATION_ENTRIES.items():
    MODEL_REGISTRY.setdefault(_key, _entry)


# 方便外部按需取得消融键名列表 (例如批量训练时)
ABLATION_KEYS = list(_ABLATION_ENTRIES.keys())


if __name__ == "__main__":
    # 自检:打印注册成功的消融条目
    print("Registered ablation entries:")
    for k in ABLATION_KEYS:
        builder, kwargs, desc = MODEL_REGISTRY[k]
        print(f"  {k:18s} | kwargs={kwargs} | {desc}")
    print(f"\nTotal models in registry now: {len(MODEL_REGISTRY)}")
