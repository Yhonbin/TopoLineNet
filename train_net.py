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
import json
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.log_dir / "history.json"
        # 若存在历史 history,加载以支持断点续训
        if self.history_path.exists():
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {'epoch': [], 'loss': [], 'loss_sup': [],
                            'loss_unsup': [], 'val_loss': []}
        
    def update(self, epoch, loss, loss_sup, loss_unsup, val_loss):
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['loss_sup'].append(loss_sup)
        self.history['loss_unsup'].append(loss_unsup)
        self.history['val_loss'].append(val_loss)
        # 实时持久化,防止训练中断丢失
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
    def plot_loss(self):
        plt.style.use('seaborn-v0_8-paper')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.history['epoch'], self.history['loss'], label='Train Total', color='black', linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_loss'], label='Validation', color='red', linewidth=2)
        ax.plot(self.history['epoch'], self.history['loss_sup'], label='Supervised', linestyle='--')
        ax.plot(self.history['epoch'], self.history['loss_unsup'], label='Unsupervised', linestyle=':')
        ax.set_title('Training and Validation Convergence', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.log_dir / "convergence_curve.pdf")
        plt.close()
        
        
class EarlyStopping:
    """
    简单早停:验证 loss 连续 patience 个 epoch 没有改善超过 min_delta,则停止
    """
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
 
    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True   # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
 
    def state_dict(self):
        return {'best_loss': self.best_loss, 'counter': self.counter,
                'should_stop': self.should_stop}
 
    def load_state_dict(self, sd):
        self.best_loss = sd['best_loss']
        self.counter = sd['counter']
        self.should_stop = sd['should_stop']


def save_checkpoint(path, model, ema_model, optimizer, scheduler, scaler,
                    early_stopper, epoch, best_val_loss):
    """
    保存完整训练状态,支持断点续训
    """
    def _state(m):
        return m.module.state_dict() if hasattr(m, 'module') else m.state_dict()
 
    ckpt = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model': _state(model),
        'ema_model': _state(ema_model) if ema_model is not None else None,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'early_stopper': early_stopper.state_dict(),
    }
    # 原子写:先写 tmp 再 rename,避免训练中断时 ckpt 写一半
    tmp_path = str(path) + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    
def load_checkpoint(path, model, ema_model, optimizer, scheduler, scaler,
                    early_stopper, device):
    """
    加载完整训练状态
    """
    ckpt = torch.load(path, map_location=device)
 
    def _load(m, sd):
        if sd is None:
            return
        if hasattr(m, 'module'):
            m.module.load_state_dict(sd, strict=False)
        else:
            m.load_state_dict(sd, strict=False)
 
    _load(model, ckpt['model'])
    if ema_model is not None:
        _load(ema_model, ckpt.get('ema_model'))
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
    early_stopper.load_state_dict(ckpt['early_stopper'])
 
    return ckpt['epoch'], ckpt['best_val_loss']


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
            raise RuntimeError("DataLoader empty")
            
def simulate_hard_occlusion(imgs, intensity=1.0):
    """遮挡强度由 intensity ∈ [0, 1] 控制，配合 consistency ramp 同步爬坡"""
    erased_imgs = imgs.clone()
    p_eff = 0.3 + 0.5 * intensity                # 0.3 -> 0.8
    num_max = max(2, int(2 + 2 * intensity))     # 2 -> 4 (上界,不含)
    size_min = 20
    size_max = int(40 + 40 * intensity)          # 40 -> 80
    for i in range(imgs.size(0)):
        if torch.rand(1).item() < p_eff:
            h, w = imgs.size(2), imgs.size(3)
            for _ in range(torch.randint(1, num_max, (1,)).item()):
                h_e = torch.randint(size_min, size_max, (1,)).item()
                w_e = torch.randint(size_min, size_max, (1,)).item()
                y1 = torch.randint(0, h - h_e, (1,)).item()
                x1 = torch.randint(0, w - w_e, (1,)).item()
                erased_imgs[i, :, y1:y1+h_e, x1:x1+w_e] = 0.9
                if torch.rand(1).item() < 0.5:
                    erased_imgs[i, :, y1+10:y1+h_e-10, x1+10:x1+w_e-10] = torch.rand(1).item() * 0.5
    return erased_imgs
            
def get_current_consistency_weight(epoch, max_epochs, max_weight=0.6, rampup_length=0.4):
    """自监督权重温和预热函数：前 30% epochs 慢慢增加到 max_weight"""
    rampup_epochs = max_epochs * rampup_length
    if epoch < rampup_epochs:
        p = max(0.0, float(epoch) / float(rampup_epochs))
        rampup_value = math.exp(-5.0 * (1.0 - p) ** 2) # 高斯预热曲线
        return max_weight * rampup_value
    return max_weight
    
# EMA 更新函数：缓慢更新教师模型
@torch.no_grad()
def update_ema_variables(model, ema_model, alpha=0.995):
    msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    esd = ema_model.module.state_dict() if hasattr(ema_model, 'module') else ema_model.state_dict()
    for k in esd.keys():
        if esd[k].dtype.is_floating_point:
            esd[k].mul_(alpha).add_(msd[k].detach(), alpha=1 - alpha)

def train_experiment():
    
    args = parse_args()
    # ------- 实验目录 -------
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            exp_dir = resume_path
            ckpt_path = exp_dir / "last_model.pth"
        else:
            ckpt_path = resume_path
            exp_dir = resume_path.parent
        assert ckpt_path.exists(), f"checkpoint 不存在: {ckpt_path}"
        print(f"[Resume] 续训目录: {exp_dir}, ckpt: {ckpt_path}")
    else:
        if args.exp_dir:
            exp_dir = Path(args.exp_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = Path(f"./exp_{timestamp}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = exp_dir / "last_model.pth"
        print(f"[New] 新建实验目录: {exp_dir}")
    # 保存运行配置
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # ------- 设备 -------
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda' if num_gpus > 0 else 'cpu')
    use_dp = (num_gpus > 1) and (not args.no_dataparallel)
    print(f"[*] GPUs: {num_gpus}, device: {device}, DataParallel: {use_dp}")
    # cudnn 性能优化:固定 input 尺寸时开启 benchmark
    torch.backends.cudnn.benchmark = True
    
     # ------- 数据 -------
    ds_lab = HarnessDataset(args.labeled_dir, augment=True)
    ds_unlab = UnlabeledHarnessDataset(args.unlabeled_dir)
    ds_val = HarnessDataset(args.val_dir, augment=False)
    assert len(ds_lab) > 0, "标注数据为空"
 
    drop_lab = len(ds_lab) >= args.bs_lab
    loader_lab = DataLoader(ds_lab, batch_size=args.bs_lab, shuffle=True,
                            drop_last=drop_lab,
                            num_workers=args.workers,
                            persistent_workers=(args.workers > 0),
                            pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.bs_val, shuffle=False,
                            num_workers=max(1, args.workers // 2),
                            persistent_workers=(args.workers > 0),
                            pin_memory=True)
 
    use_semi = len(ds_unlab) > 0
    
    #------------ 关闭无标注数据------------#
    # use_semi = False
    
    if use_semi:
        loader_unlab = DataLoader(ds_unlab, batch_size=args.bs_unlab, shuffle=True,
                                  drop_last=True,
                                  num_workers=max(1, args.workers // 2),
                                  persistent_workers=(args.workers > 0),
                                  pin_memory=True)
        iter_unlab = iter(cycle(loader_unlab))
    print(f"[*] labeled={len(ds_lab)}, unlabeled={len(ds_unlab)}, val={len(ds_val)}")
    print(f"[*] len(loader_lab)={len(loader_lab)}, "
          f"len(loader_unlab)={len(loader_unlab) if use_semi else 0}")

        
    # ------- 模型 -------
    model = HarnessHRNetV2(pretrained=(args.resume is None)).to(device)
    if use_semi:
        ema_model = HarnessHRNetV2(pretrained=False).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
    else:
        ema_model = None
 
    if use_dp:
        model = nn.DataParallel(model)
        # 教师模型 batch 小,放单卡反而更快;此处不包 DataParallel
        
    # ------- 优化器 / 调度器 / AMP -------
    # 学习率不再乘以 GPU 数量,因为 batch_size 没有等比增大
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base_lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)     
   
    # ------- 一个 epoch 真正合理的 step 数 -------
    if use_semi:
        # 让数据量大的一侧每个 epoch 至少完整过一遍
        iterations_per_epoch = max(len(loader_lab), len(loader_unlab))
    else:
        iterations_per_epoch = len(loader_lab)
    print(f"[*] iterations_per_epoch = {iterations_per_epoch}")
    
    # ------- Logger / 早停 -------
    logger = ExperimentLogger(exp_dir)
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    # ------- Resume -------
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            ckpt_path, model, ema_model, optimizer, scheduler, scaler,
            early_stopper, device)
        start_epoch += 1
        print(f"[Resume] 从 epoch {start_epoch} 继续, best_val_loss={best_val_loss:.4f}")
 
    iter_lab = iter(cycle(loader_lab))

    # 2. Loop
    print(f"Starting Semi-Supervised Training | Labeled: {len(ds_lab)} | Unlabeled: {len(ds_unlab)}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if ema_model is not None:
            ema_model.eval()  # 教师不开 dropout/BN train
        total_loss = total_sup = total_unsup = 0.0
        consist_w = get_current_consistency_weight(
            epoch, args.epochs, args.max_consist_w, args.rampup)
 

        # 使用 tqdm 进度条展示每一轮内部的进度
        pbar = tqdm(range(iterations_per_epoch),desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for _ in pbar:
            optimizer.zero_grad(set_to_none=True)
            # ---------------------------------------------------------
            # Step 1: 有监督学习 (Supervised on Labeled Data)
            # ---------------------------------------------------------
            imgs_lab, t_line = next(iter_lab)
            imgs_lab, t_line = imgs_lab.to(device, non_blocking=True), t_line.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                p_line = model(imgs_lab)
                if p_line.shape[-2:] != t_line.shape[-2:]:
                    t_line = F.interpolate(t_line, size=p_line.shape[-2:],
                                           mode='bilinear', align_corners=False)
                supervised_loss = harness_topology_loss(p_line.float(), t_line)  
            
                # ---------------------------------------------------------
                # Step 2: 自监督一致性学习 (Self-Supervised on Unlabeled Data)
                # ---------------------------------------------------------
                consistency_loss = torch.tensor(0.0,device=device)
                if use_semi:
                    imgs_unlab = next(iter_unlab).to(device, non_blocking=True)
                    with torch.no_grad():
                        pseudo = ema_model(imgs_unlab).detach() # 教师模型直接给出指导目标
                        
                    # 学生模型：输入被严重遮挡的图片，试图还原教师的预测结果
                    imgs_unlab_strong = simulate_hard_occlusion(imgs_unlab, intensity=consist_w / max(args.max_consist_w, 1e-6))
                    p_student = model(imgs_unlab_strong)
                    if p_student.shape != pseudo.shape:
                            pseudo = F.interpolate(pseudo, size=p_student.shape[-2:],
                                                mode='bilinear', align_corners=False)
                    # 计算一致性损失 (学生要在遮挡下猜出老师的答案)
                    consistency_loss = F.mse_loss(p_student.float(), pseudo.float()) * 10.0
                
                loss = supervised_loss + consist_w * consistency_loss
            # AMP 反传
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 更新教师模型权重
            if use_semi:
                update_ema_variables(model, ema_model, alpha=args.ema_alpha)
            
            total_loss += loss.item()
            total_sup += supervised_loss.item()
            if use_semi:
                total_unsup += consistency_loss.item()
            
            # 更新进度条右侧的显示
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sup': f"{supervised_loss.item():.4f}",
                'unsup': f"{consistency_loss.item():.4f}",
                'cw': f"{consist_w:.2f}",
            })
        # 更新学习率
        scheduler.step()
        # 记录本 Epoch 平均 Loss
        avg_loss = total_loss / iterations_per_epoch
        avg_sup = total_sup / iterations_per_epoch
        avg_unsup = (total_unsup / iterations_per_epoch) if use_semi else 0.0

        # ------------验证集评估------------#
        if epoch % args.val_every == 0:
            val_loss = validate(model, loader_val, device, use_amp=args.amp)
        else:
            val_loss = logger.history['val_loss'][-1] if logger.history['val_loss'] \
                else float('inf')
 
        logger.update(epoch, avg_loss, avg_sup, avg_unsup, val_loss)
        print(f"[Epoch {epoch:3d}] train={avg_loss:.4f} (sup={avg_sup:.4f}, "
              f"unsup={avg_unsup:.4f}) | val={val_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | cw={consist_w:.2f}")
        
        # 可视化
        if epoch % args.vis_every == 0:
            save_snapshot(imgs_lab[0], t_line[0], p_line[0],
                          epoch, exp_dir, prefix="lab")
            if use_semi:
                save_snapshot(imgs_unlab[0], None, pseudo[0],
                              epoch, exp_dir, prefix="unlab",
                              aug_image=imgs_unlab_strong[0])
       
        # 保存 best
        improved = early_stopper.step(val_loss)
        if improved:
            best_val_loss = val_loss
            best_path = exp_dir / "best_model.pth"
            state = model.module.state_dict() if hasattr(model, 'module') \
                else model.state_dict()
            torch.save(state, best_path)
            print(f"   ↑ best updated: {best_val_loss:.4f}")
            
        # 保存 last(支持断点续训)
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(exp_dir / "last_model.pth",
                            model, ema_model, optimizer, scheduler, scaler,
                            early_stopper, epoch, best_val_loss)
        
        # 早停
        if early_stopper.should_stop:
            print(f"[EarlyStop] val_loss 在 {args.patience} 个 epoch 内未改善,停止")
            # 早停时也保存一份 last
            save_checkpoint(exp_dir / "last_model.pth",
                            model, ema_model, optimizer, scheduler, scaler,
                            early_stopper, epoch, best_val_loss)
            break

                
    # 3. Finalize
    logger.plot_loss()
    print(f"[Done] Best val_loss = {best_val_loss:.4f}, results in {exp_dir}")
    
    
# =====================================================
# 验证 / 可视化
# =====================================================
@torch.no_grad()
def validate(model, val_loader, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    n = 0
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    for imgs, targets in val_loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda",enabled=use_amp):
            preds = model(imgs)
            loss = harness_topology_loss(preds.float(), targets)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / max(1, n)
    
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
    
    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labeled_dir',   default='./data/train')
    p.add_argument('--unlabeled_dir', default='./data/data_unlabeled')
    p.add_argument('--val_dir',       default='./data/val')
 
    # 关键:resume 控制断点续训
    p.add_argument('--resume', default=None,
                   help='路径:已有实验目录(包含 last_model.pth)或 .pth 文件;'
                        '若指定,则在该目录续训')
    p.add_argument('--exp_dir', default=None,
                   help='可选:手动指定实验输出目录,不指定则用时间戳新建')
 
    # 训练超参
    p.add_argument('--bs_lab',   type=int,   default=8)    # ← 改小
    p.add_argument('--bs_unlab', type=int,   default=4)    # ← 改小
    p.add_argument('--bs_val',   type=int,   default=8)
    p.add_argument('--epochs',   type=int,   default=150)  # ← 缩短
    p.add_argument('--base_lr',  type=float, default=2e-4) # ← 调小
    p.add_argument('--wd',       type=float, default=5e-4) # ← 调小
    p.add_argument('--max_consist_w', type=float, default=0.5)
    p.add_argument('--rampup',   type=float, default=0.4)
    p.add_argument('--ema_alpha', type=float, default=0.995)
 
    # 工程相关
    p.add_argument('--workers',  type=int, default=2)      # ← 数据少不需要 4
    p.add_argument('--amp',      action='store_true', default=True)
    p.add_argument('--val_every', type=int, default=2,
                   help='每 N 个 epoch 跑一次 validate')
    p.add_argument('--vis_every', type=int, default=10)
    p.add_argument('--ckpt_every', type=int, default=5,
                   help='每 N 个 epoch 保存一次 last_model')
 
    # 早停
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--min_delta', type=float, default=1e-4)
 
    # DDP / DP
    p.add_argument('--no_dataparallel', action='store_true',
                   help='禁用 DataParallel,单卡训练')
 
    return p.parse_args()



    
if __name__ == "__main__":
    from contextlib import nullcontext
    train_experiment()
    # evaluate_only(model_path="./Mean-Teacher/best_model.pth", val_dir="./data/val", save_dir="./val_results")