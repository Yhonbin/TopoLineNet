import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from Datasets import HarnessDataset,UnlabeledHarnessDataset
from HRNet import HarnessHRNetV2
from tqdm import tqdm
import math
from loss import hybrid_loss_fn, harness_topology_loss
from skimage.morphology import skeletonize
from utils import smooth_and_skeletonize
from datetime import datetime


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 根据你的 GPU 数量调整

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {'epoch': [], 'loss': [], 'loss_sup': [], 'loss_unsup': [], 'val_loss': []}
        
    def update(self, epoch, loss, loss_sup, loss_unsup,val_loss):
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['loss_sup'].append(loss_sup)
        self.history['loss_unsup'].append(loss_unsup)
        self.history['val_loss'].append(val_loss)
        
    def plot_loss(self):
        plt.style.use('seaborn-v0_8-paper')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 绘制训练总损失（黑色实线）
        ax.plot(self.history['epoch'], self.history['loss'], label='Train Total Loss', color='black', linewidth=2)
        # 绘制验证总损失（红色实线，醒目对比）
        ax.plot(self.history['epoch'], self.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        
        ax.plot(self.history['epoch'], self.history['loss_sup'], label='Supervised', linestyle='--')
        ax.plot(self.history['epoch'], self.history['loss_unsup'], label='Unsupervised', linestyle=':')
        ax.set_title('Training and Validation Convergence', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(); plt.tight_layout()
        plt.savefig(self.log_dir / "convergence_curve.pdf"); plt.close()
        
# ==========================================
# Module 4: Execution Entry
# ==========================================

def cycle(iterable):
    """无限迭代器：用于平衡大小不同的数据集。修复了 DataLoader 为空时的死循环问题"""
    while True:
        is_empty = True
        for x in iterable:
            is_empty = False
            yield x
        if is_empty:
            raise RuntimeError(
                "\n[致命错误] DataLoader 为空！程序已拦截死循环。\n"
                "原因分析：可能你的数据文件夹中图片少于 batch_size (当前为2)，"
                "且 DataLoader 设置了 drop_last=True 导致所有数据被丢弃。\n"
                "请检查 './data_labeled' 或 './data_unlabeled' 文件夹内的文件数量。"
            )
            
def simulate_hard_occlusion(imgs, p=0.8):
    """数据增强：强迫模型学习在遮挡下推理连通性"""
    erased_imgs = imgs.clone()
    for i in range(imgs.size(0)):
        if torch.rand(1).item() < p:
            h, w = imgs.size(2), imgs.size(3)
            # 模拟贴有白底黑字的标签
            for _ in range(torch.randint(1, 4, (1,)).item()):
                h_e, w_e = torch.randint(30, 80, (1,)).item(), torch.randint(30, 80, (1,)).item()
                y1, x1 = torch.randint(0, h - h_e, (1,)).item(), torch.randint(0, w - w_e, (1,)).item()
                erased_imgs[i, :, y1:y1+h_e, x1:x1+w_e] = 0.9 # 白色
                # 模拟标签上的条形码/文字
                if torch.rand(1).item() < 0.5:
                    erased_imgs[i, :, y1+10:y1+h_e-10, x1+10:x1+w_e-10] = torch.rand(1).item() * 0.5
    return erased_imgs
            
def get_current_consistency_weight(epoch, max_epochs=150, max_weight=0.8, rampup_length=0.3):
    """自监督权重温和预热函数：前 30% epochs 慢慢增加到 max_weight"""
    rampup_epochs = max_epochs * rampup_length
    if epoch < rampup_epochs:
        p = max(0.0, float(epoch) / float(rampup_epochs))
        rampup_value = math.exp(-5.0 * (1.0 - p) ** 2) # 高斯预热曲线
        return max_weight * rampup_value
    else:
        return max_weight
    
# EMA 更新函数：缓慢更新教师模型
def update_ema_variables(model, ema_model, alpha=0.99):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train_experiment():
    # 1. Init
    LABELED_DIR = "./data/train"     # 存放 10 张图 + json 的文件夹.训练集文件夹
    UNLABELED_DIR = "./data/data_unlabeled" # 存放几十/上百张没标注的图片的文件夹
    VAL_DIR = "./data/val"           # 存放验证集的文件夹，结构同 train
    # DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RES_DIR = os.path.join('.', f"exp_{timestamp}") 
    
    # 检测 GPU 数量
    num_gpus = torch.cuda.device_count()
    DEVICE = torch.device('cuda' if num_gpus > 0 else 'cpu')
    print(f"[*] Detected {num_gpus} GPUs. Using device: {DEVICE}")
    # 如果检测到多张卡，使用 DataParallel 包装模型
    
    dataset_lab = HarnessDataset(LABELED_DIR, augment=True)
    dataset_unlab = UnlabeledHarnessDataset(UNLABELED_DIR)
    dataset_val = HarnessDataset(VAL_DIR, augment=False) 
    if len(dataset_lab) == 0: exit("Error: No labeled data found.")
    drop_lab = len(dataset_lab) >= 2
    loader_lab = DataLoader(dataset_lab, batch_size=8, shuffle=True, drop_last=drop_lab)
    loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False)
    
    # 如果有无标签数据，则开启半监督模式
    use_semi_supervised = len(dataset_unlab) > 0
    
    #------------ 关闭无标注数据------------#
    # use_semi_supervised = False
    
    if use_semi_supervised:
        loader_unlab = DataLoader(dataset_unlab, batch_size=8, shuffle=True, drop_last=drop_lab)
        iter_unlab = iter(cycle(loader_unlab)) # 
        
    # 1. 学生模型 (被优化器更新)
    model = HarnessHRNetV2(pretrained=True).to(DEVICE)
    # 2. 教师模型 (不参与反向传播，仅通过 EMA 更新)
    ema_model = HarnessHRNetV2(pretrained=False).to(DEVICE)
    ema_model.load_state_dict(model.state_dict()) # 初始化权重相同
    for param in ema_model.parameters():
        param.detach_()
    if num_gpus > 1:
        print(f"[*] Multi-GPU mode enabled. Using {num_gpus} GPUs with DataParallel.")
        model = nn.DataParallel(model)
        ema_model =nn.DataParallel(ema_model)
    base_lr = 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr * max(1, num_gpus // 2), weight_decay=1e-3)
    MAX_EPOCHS = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    
    logger = ExperimentLogger(RES_DIR)

    # 定义一个 Epoch 的长度：取标注数据和未标注数据中 Batch 较多的那个
    iterations_per_epoch = len(loader_lab) * 2 if use_semi_supervised else len(loader_lab)
    # 提取有标签的数据
    iter_lab = iter(cycle(loader_lab))
    
    best_val_loss = float('inf') # 用于跟踪最佳验证损失，保存最优模型

    # 2. Loop
    print(f"Starting Semi-Supervised Training | Labeled: {len(dataset_lab)} | Unlabeled: {len(dataset_unlab)}")

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        total_loss, total_sup, total_unsup = 0, 0, 0
        current_consist_weight = get_current_consistency_weight(epoch, max_epochs=MAX_EPOCHS, max_weight=1.0)

        # 使用 tqdm 进度条展示每一轮内部的进度
        pbar = tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch}/{MAX_EPOCHS}", unit="batch")
        for _ in pbar:
            optimizer.zero_grad()
            # ---------------------------------------------------------
            # Step 1: 有监督学习 (Supervised on Labeled Data)
            # ---------------------------------------------------------
            imgs_lab, t_line = next(iter_lab)
            imgs_lab, t_line = imgs_lab.to(DEVICE), t_line.to(DEVICE)
            
            # ------------在有标签的数据上直接贴白块，让带有 GT 的数据教网络跨越障碍！------------
            # imgs_lab_aug = simulate_hard_occlusion(imgs_lab, p=0.7)
            # -----------------------
            p_line = model(imgs_lab)
            supervised_loss = harness_topology_loss(p_line, t_line)  
            
            # ---------------------------------------------------------
            # Step 2: 自监督一致性学习 (Self-Supervised on Unlabeled Data)
            # ---------------------------------------------------------
            consistency_loss = torch.tensor(0.0).to(DEVICE)
            
            if use_semi_supervised:
                imgs_unlab = next(iter_unlab).to(DEVICE)
                with torch.no_grad():
                    pseudo_labels = ema_model(imgs_unlab).detach() # 教师模型直接给出指导目标
                    
                # 学生模型：输入被严重遮挡的图片，试图还原教师的预测结果
                imgs_unlab_strong = simulate_hard_occlusion(imgs_unlab, p=0.8)
                p_unlab_student = model(imgs_unlab_strong)
                
                # 计算一致性损失 (学生要在遮挡下猜出老师的答案)
                consistency_loss = nn.MSELoss()(p_unlab_student, pseudo_labels) * 10.0
                
            # ---------------------------------------------------------
            # Step 3: 联合优化
            # ---------------------------------------------------------
            loss = supervised_loss + current_consist_weight * consistency_loss
            loss.backward()
            optimizer.step()
            
            # 更新教师模型权重 (动量 0.99)
            if use_semi_supervised:
                update_ema_variables(model, ema_model, alpha=0.99)
            
            total_loss += loss.item()
            total_sup += supervised_loss.item()
            if use_semi_supervised: total_unsup += consistency_loss.item()
            
            # 更新进度条右侧的显示
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sup': f"{supervised_loss.item():.4f}",
                'unsup': f"{consistency_loss.item():.4f}"
            })
        # 更新学习率
        scheduler.step()
        # 记录本 Epoch 平均 Loss
        avg_loss = total_loss / iterations_per_epoch
        avg_sup = total_sup / iterations_per_epoch
        avg_unsup = (total_unsup / iterations_per_epoch) if use_semi_supervised else 0

        # ------------验证集评估------------#
        val_loss = validate(model, loader_val, DEVICE)
        logger.update(epoch, avg_loss, avg_sup, avg_unsup, val_loss)
        

        if epoch % 10 == 0:
            print(f"[*] Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")
            save_snapshot(imgs_lab[0], t_line[0], p_line[0], epoch, RES_DIR, prefix="lab")
            if use_semi_supervised:
                save_snapshot(imgs_unlab[0], None, pseudo_labels[0], epoch, RES_DIR, prefix="unlab",aug_image=imgs_unlab_strong[0])
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"{RES_DIR}/best_model.pth"
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
                
    # 3. Finalize
    logger.plot_loss()
    print(f"Experiment finished. Best Val Loss: {best_val_loss:.4f}. Reports generated in {RES_DIR}")
    
    
def validate(model, val_loader, device):
    """在验证集上计算指标，关闭梯度更新"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            
            # 使用同样的拓扑损失函数评估
            loss = harness_topology_loss(preds, targets)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)
    
def save_snapshot(image, t_line, p_line, epoch, save_dir, prefix="lab", aug_image=None):
    img_np = image.permute(1, 2, 0).cpu().numpy()
    output_np = p_line.detach().cpu().numpy().squeeze()
    
    skel_visual = smooth_and_skeletonize(output_np, threshold=0.5)
    
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    if aug_image is not None:
        plt.imshow(aug_image.permute(1, 2, 0).cpu().numpy().squeeze()); plt.title("Simulated Occlusion")
    else:
        plt.imshow(img_np); plt.title("Original Input")
        
    plt.subplot(1, 4, 2)
    if t_line is not None:
        plt.imshow(t_line.cpu().numpy().squeeze(), cmap='jet'); plt.title("Target GT")
    else:
        plt.text(0.5, 0.5, "Unlabeled", ha='center', va='center'); plt.title("No GT")
        
    plt.subplot(1, 4, 3)
    plt.imshow(output_np, cmap='jet'); plt.title("Raw Heatmap (Band)")
    
    plt.subplot(1, 4, 4)
    plt.imshow(skel_visual, cmap='gray'); plt.title("Skeletonized Centerline (Smooth)")
    
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"{prefix}_epoch_{epoch}.png")); plt.close()
    
    

# ==========================================
# 新增：独立验证与可视化 (Standalone Evaluation)
# ==========================================
def evaluate_only(model_path, val_dir, save_dir="./val_results"):
    """直接调用此函数进行验证集测试和可视化"""
    os.makedirs(save_dir, exist_ok=True)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化并加载最优模型
    model = HarnessHRNetV2(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    val_dataset = HarnessDataset(val_dir, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"[*] 开始验证，共 {len(val_dataset)} 张测试图...")
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(tqdm(val_loader)):
            imgs, targets = imgs.to(device=DEVICE), targets.to(device=DEVICE)
            preds = model(imgs)
            
            # 转为 numpy 准备画图
            img_np = imgs[0].permute(1, 2, 0).cpu().numpy()
            gt_np = targets[0].cpu().numpy().squeeze()
            pred_np = preds[0].cpu().numpy().squeeze()
            
            # 后处理平滑与骨架化
            skel_np = smooth_and_skeletonize(pred_np, threshold=0.5)
            
            # 简单可视化保存
            plt.figure(figsize=(16, 4))
            plt.subplot(1,4,1); plt.imshow(img_np); plt.title("Input")
            plt.subplot(1,4,2); plt.imshow(gt_np, cmap='jet'); plt.title("GT Ribbon")
            plt.subplot(1,4,3); plt.imshow(pred_np, cmap='jet'); plt.title("Pred Heatmap")
            plt.subplot(1,4,4); plt.imshow(skel_np, cmap='gray'); plt.title("Skeleton")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/val_result_{i}.png")
            plt.close()
            
    print(f"[*] 验证结束！可视化结果已保存至 {save_dir}")
    
    
if __name__ == "__main__":
    train_experiment()
    
    # evaluate_only(model_path="./Mean-Teacher/best_model.pth", val_dir="./data/val", save_dir="./val_results")