import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FocalLoss(nn.Module):
    """解决 99% 背景与 1% 线条的极端类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        # alpha 权重倾向于正样本 (线束)
        loss = -self.alpha * target * torch.log(pred) * (1 - pred) ** self.gamma \
               - (1 - self.alpha) * (1 - target) * torch.log(1 - pred) * pred ** self.gamma
        return loss.mean()


def consistency_dice_loss(p_clean, p_aug):
    """自监督一致性：让遮挡后的预测看起来像没遮挡的图"""
    p1 = p_clean.reshape(-1)
    p2 = p_aug.reshape(-1)
    intersection = (p1 * p2).sum()
    union = p1.sum() + p2.sum()
    if union < 1e-5: return torch.tensor(0.0).to(p_clean.device)
    return 1 - (2. * intersection + 1e-5) / (union + 1e-5)


# BCE + Dice Loss 的混合版本，给断裂处更高权重，同时保持整体结构
def hybrid_loss_fn(pred, target, bce_w=1.0, dice_w=1.5):
    """计算单个分支的 BCE + Dice Loss"""
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    bce = nn.BCELoss()(pred, target)
    p_flat, t_flat = pred.view(-1), target.view(-1)
    dice = (2. * (p_flat * t_flat).sum() + 1e-5) / ((p_flat + t_flat).sum() + 1e-5)
    # 引入 clDice (可以适当调整权重)
    # cldice = cldice_loss(pred, target)
    
    return bce_w * bce + dice_w * (1 - dice)


def soft_erode(img):
    """使用 max pool 模拟形态学腐蚀"""
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    """使用 max pool 模拟形态学膨胀"""
    return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_skeletonize(img, iter_=3):
    """软骨架提取，用于计算拓扑连通性"""
    skel = torch.zeros_like(img)
    for _ in range(iter_):
        eroded = soft_erode(img)
        temp = soft_dilate(eroded)
        # img - temp 提取边缘/骨架信息
        skel = torch.max(skel, F.relu(img - temp)) 
        img = eroded
    return skel

def soft_cldice_loss(pred, target):
    """Centerline Dice: 保证网络预测出的线束能够连通，不断裂"""
    pred_skel = soft_skeletonize(pred)
    target_skel = soft_skeletonize(target)
    
    # 交叉验证骨架与原始分割的重合度
    tprec = (torch.sum(pred_skel * target) + 1e-5) / (torch.sum(pred_skel) + 1e-5)
    tsens = (torch.sum(target_skel * pred) + 1e-5) / (torch.sum(target_skel) + 1e-5)
    
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
    return 1.0 - cl_dice


def harness_topology_loss(pred, target):
    """混合损失：Focal 负责像素级找线，clDice 负责拓扑连通"""
    focal = FocalLoss()(pred, target)
    cldice = soft_cldice_loss(pred, target)
    return 10.0 * focal + 2.0 * cldice