import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import math
# ==========================================
# Module 1: Models (HRNet + CBAM + Strip Pooling)
# ==========================================
# class CBAMLayer(nn.Module):
#     """CBAM 注意力机制：提升模型在复杂背景（如白色标签）下的辨别力"""
#     def __init__(self, channels, reduction=16):
#         super(CBAMLayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = self.sigmoid(avg_out + max_out)
#         return x * out
    
    
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
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        # 辅助函数:统一构建 Conv + GN + ReLU
        def _gn(c):
            # 保证 num_groups 能整除 channels
            g = num_groups if c % num_groups == 0 else math.gcd(num_groups, c)
            return nn.GroupNorm(g, c)

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
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

class HarnessHRNetV2(nn.Module):
    def __init__(self, model_name='hrnet_w18', pretrained=False):
        super(HarnessHRNetV2, self).__init__()
        self.backbone = timm.create_model(model_name,features_only=True)
        if (pretrained):
            self.backbone.load_state_dict(torch.load('./pretrained_model/hrnetv2_w18_imagenet_pretrained.pth'), strict=False)
        feature_info = self.backbone.feature_info.channels()
        
        # 增加注意力模块
        self.attentions = nn.ModuleList([CBAMLayer(c) for c in feature_info])
        self.strip_pools = nn.ModuleList([StripPooling(c) for c in feature_info])
        total_channels = sum(feature_info)
        aspp_out_channels = 256
        
        self.neck = ASPP(total_channels, aspp_out_channels)  # 新增 Neck 模块，输出固定通道数
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

    def forward(self, x):
        features = self.backbone(x)
        # 应用注意力
        # features = [self.attentions[i](f) for i, f in enumerate(features)]
        # target_size = features[0].shape[-2:]
        # resized_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=True) for f in features]
        enhanced_features = []
        for i, f in enumerate(features):
            f_att = self.attentions[i](f)
            f_strip = self.strip_pools[i](f_att) # 捕获被标签遮挡的上下文
            enhanced_features.append(f_strip)
        target_size = enhanced_features[0].shape[-2:]
        resized_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=True) for f in enhanced_features]
        combined = torch.cat(resized_features, dim=1)
        
        neck_out = self.neck(combined)  # 通过 Neck 模块融合多尺度特征
        # 输出
        out_line = torch.sigmoid(F.interpolate(self.head_line(neck_out), size=x.shape[-2:], mode='bilinear', align_corners=True))
    
        return out_line

