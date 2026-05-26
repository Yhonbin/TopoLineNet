import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import math
# ==========================================
# Module 1: Models (HRNet + CBAM + Strip Pooling)
# ==========================================

class CBAMLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        x = x * self.sigmoid(avg_out + max_out)
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial(spatial)
        x = x * spatial
        return x

class StripPooling(nn.Module): 
    """条形池化：跨越白色标签遮挡的核心"""
    def __init__(self, in_channels, pool_size=(256, 256)):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, pool_size[1])) 
        self.pool2 = nn.AdaptiveAvgPool2d((pool_size[0], 1)) 
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool_h = F.interpolate(self.conv1_1(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        pool_w = F.interpolate(self.conv1_2(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        return x * self.conv2(pool_h + pool_w)
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32, dilations=(6,12)):
        super().__init__()
        # 辅助函数:统一构建 Conv + GN + ReLU
        def _gn(c):
            # 保证 num_groups 能整除 channels
            g = num_groups if c % num_groups == 0 else math.gcd(num_groups, c)
            return nn.GroupNorm(g, c)
        
        d1, d2 = dilations  # 暴露空洞率，方便消融时改小（如 (2,4)）验证细结构友好性

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=d1, dilation=d1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=d2, dilation=d2, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True)
        )
        # global_pool 分支:这里是问题点,必须用 GN
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = F.interpolate(self.global_pool(x), size=x.size()[2:],
                           mode='bilinear', align_corners=True)
        return self.project(torch.cat((x1, x2, x3, x4), dim=1))
    

class SimpleNeck(nn.Module):
    """关闭 ASPP 时的替代 neck：仅做通道融合/压缩，不引入多空洞分支。"""
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
 
        def _gn(c):
            g = num_groups if c % num_groups == 0 else math.gcd(num_groups, c)
            return nn.GroupNorm(g, c)
 
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)    


class HarnessHRNetV2(nn.Module):
    """
    带消融开关的主干模型。
 
    消融开关 (默认全开 -> 行为与原始模型完全一致):
        use_cbam        : 是否启用 CBAM 通道+空间注意力
        use_strip_pool  : 是否启用 StripPooling 条形池化
        use_aspp        : 是否启用 ASPP neck (关闭时用 SimpleNeck 替代)
 
    设计要点
    --------
    1. 关闭某模块时是「真正跳过该模块的前向计算」(用 nn.Identity 或
       完全不构建)，而不是乘以 0，因此 profile_models.py 测到的耗时变化
       是真实的，可直接作为消融实验的速度数据。
    2. head 的输入通道恒为 aspp_out_channels，无论 ASPP 开关如何，下游
       (train_net.py / compare_models.py / evaluate_metric.py) 全部无需改动。
    3. 关闭模块时其参数不会被构建，故消融配置下的参数量统计也是真实的。
 
    用法
    ----
        # 完整模型 (= 原始行为)
        model = HarnessHRNetV2(pretrained=True)
 
        # 只去掉 ASPP
        model = HarnessHRNetV2(pretrained=True, use_aspp=False)
 
        # 只保留 backbone (= vanilla 风格, 三个模块全关)
        model = HarnessHRNetV2(pretrained=True,
                               use_cbam=False, use_strip_pool=False, use_aspp=False)
    """
    def __init__(self, model_name='hrnet_w18', pretrained=False,
                 use_cbam=True, use_strip_pool=True, use_aspp=True, aspp_dilations=(6,12)):
        super(HarnessHRNetV2, self).__init__()
        # ---- 记录消融配置 ----
        self.use_cbam = use_cbam
        self.use_strip_pool = use_strip_pool
        self.use_aspp = use_aspp

        self.backbone = timm.create_model(model_name,features_only=True)
        if (pretrained):
            self.backbone.load_state_dict(torch.load('./pretrained_model/hrnetv2_w18_imagenet_pretrained.pth'), strict=False)
        feature_info = self.backbone.feature_info.channels()
        
        # ---- 注意力模块 (按开关构建; 关闭则用 Identity 占位以保持索引一致) ----
        if self.use_cbam:
            self.attentions = nn.ModuleList([CBAMLayer(c) for c in feature_info])
        else:
            self.attentions = nn.ModuleList([nn.Identity() for _ in feature_info])
 
        if self.use_strip_pool:
            self.strip_pools = nn.ModuleList([StripPooling(c) for c in feature_info])
        else:
            self.strip_pools = nn.ModuleList([nn.Identity() for _ in feature_info])
            
        total_channels = sum(feature_info)
        aspp_out_channels = 256
        
        # ---- Neck: ASPP 或 轻量替代 ----
        if self.use_aspp:
            self.neck = ASPP(total_channels, aspp_out_channels, dilations=aspp_dilations)
        else:
            self.neck = SimpleNeck(total_channels, aspp_out_channels)
        # Head A: 预测线束中心线
        self.head_line = nn.Sequential(
            nn.Conv2d(aspp_out_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # H/4 -> H/2
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # H/2 -> H
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1) # 输出单通道热力图
        )
    
    def ablation_tag(self):
        """返回当前消融配置的可读标签，便于日志/权重命名。"""
        parts = []
        parts.append("cbam" if self.use_cbam else "noCbam")
        parts.append("strip" if self.use_strip_pool else "noStrip")
        parts.append("aspp" if self.use_aspp else "noAspp")
        return "_".join(parts)

    def forward(self, x):
        features = self.backbone(x)
 
        enhanced_features = []
        for i, f in enumerate(features):
            # CBAM: 开关关闭时 self.attentions[i] 是 Identity, 零开销跳过
            f_att = self.attentions[i](f)
            # StripPooling: 同上
            f_strip = self.strip_pools[i](f_att)
            enhanced_features.append(f_strip)
 
        target_size = enhanced_features[0].shape[-2:]
        resized_features = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=True)
            for f in enhanced_features
        ]
        combined = torch.cat(resized_features, dim=1)
 
        neck_out = self.neck(combined)  # ASPP 或 SimpleNeck
 
        out_line = torch.sigmoid(
            F.interpolate(self.head_line(neck_out), size=x.shape[-2:],
                          mode='bilinear', align_corners=True))
 
        return out_line

