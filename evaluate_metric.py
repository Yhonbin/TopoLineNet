import torch
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# 导入你的网络和数据集定义
from utils import smooth_and_skeletonize

# ==========================================
# 1. 评价指标计算核心函数
# ==========================================

def calculate_cldice(pred_skel, gt_skel):
    """
    计算 Centerline Dice (clDice) - 衡量拓扑连通性的核心指标
    要求输入为 0-1 二值化的 numpy 数组 (单通道骨架)
    """
    pred_skel = pred_skel > 0
    gt_skel = gt_skel > 0
    
    tprec = (np.sum(pred_skel & gt_skel) + 1e-5) / (np.sum(pred_skel) + 1e-5)
    tsens = (np.sum(pred_skel & gt_skel) + 1e-5) / (np.sum(gt_skel) + 1e-5)
    
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
    return cl_dice

def calculate_relaxed_metrics(pred_skel, gt_skel, tolerance=3):
    """
    计算容差 Precision, Recall 和 F1-Score。
    因为人工标注和网络预测的中心线极难做到像素级100%重合，
    只要预测点在真实点的 `tolerance` 像素范围内，我们就认为它预测正确了。
    """
    # 计算预测骨架和真实骨架的距离变换矩阵
    # 距离变换会计算背景中每个像素到最近的非零(骨架)像素的距离
    gt_dist = cv2.distanceTransform((1 - gt_skel).astype(np.uint8), cv2.DIST_L2, 0)
    pred_dist = cv2.distanceTransform((1 - pred_skel).astype(np.uint8), cv2.DIST_L2, 0)
    
    # Precision: 预测出的点，有多少落在 GT 的容差范围内
    true_positives_p = np.sum((pred_skel > 0) & (gt_dist <= tolerance))
    precision = (true_positives_p + 1e-5) / (np.sum(pred_skel > 0) + 1e-5)
    
    # Recall: GT 中的点，有多少被预测出的点覆盖了（在容差范围内）
    true_positives_r = np.sum((gt_skel > 0) & (pred_dist <= tolerance))
    recall = (true_positives_r + 1e-5) / (np.sum(gt_skel > 0) + 1e-5)
    
    # F1-Score
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-5)
    
    return precision, recall, f1_score


def calculate_breakage_rate(pred_skel, gt_skel, tolerance=3):
    """
    断裂率 (Breakage Rate, BR)：GT 骨架连续段中，预测骨架发生断裂的比例。
    对 GT 骨架做 connected components，检查每条连续段是否被预测骨架连续覆盖。
    """
    num_components, gt_labels = cv2.connectedComponents(gt_skel.astype(np.uint8))
    if num_components == 0:
        return 0.0

    # 预测骨架的距离变换（用于容差匹配）
    pred_dist = cv2.distanceTransform((1 - pred_skel).astype(np.uint8), cv2.DIST_L2, 0)

    broken_count = 0
    for label_id in range(1, num_components + 1):
        gt_segment = (gt_labels == label_id)
        # 检查该段中是否有预测骨架覆盖（在容差范围内）
        covered = np.sum(gt_segment & (pred_dist <= tolerance))
        total = np.sum(gt_segment)
        # 如果覆盖率低于 80%，认为该段发生断裂
        if total > 0 and (covered / total) < 0.8:
            broken_count += 1

    return broken_count / num_components


def calculate_connectivity_rate(pred_skel, gt_skel, tolerance=5):
    """
    连通率 (Connectivity Rate, CR)：预测骨架端点与 GT 端点的匹配比例。
    端点定义：邻域（3x3）中非零像素数 = 1 的骨架像素。
    """
    def find_endpoints(skel):
        """提取骨架端点"""
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
        endpoints = np.argwhere((skel > 0) & (neighbor_count == 1))
        return endpoints  # shape: (N, 2), 每行是 (row, col)

    pred_endpoints = find_endpoints(pred_skel)
    gt_endpoints = find_endpoints(gt_skel)

    if len(gt_endpoints) == 0:
        return 1.0 if len(pred_endpoints) == 0 else 0.0
    if len(pred_endpoints) == 0:
        return 0.0

    # 用距离变换做容差匹配
    gt_dist = cv2.distanceTransform((1 - gt_skel).astype(np.uint8), cv2.DIST_L2, 0)
    pred_dist = cv2.distanceTransform((1 - pred_skel).astype(np.uint8), cv2.DIST_L2, 0)

    # 预测端点中，有多少在 GT 端点的容差范围内
    matched_pred = 0
    for ep in pred_endpoints:
        r, c = ep
        if gt_dist[r, c] <= tolerance:
            matched_pred += 1

    # GT 端点中，有多少在预测端点的容差范围内
    matched_gt = 0
    for ep in gt_endpoints:
        r, c = ep
        if pred_dist[r, c] <= tolerance:
            matched_gt += 1

    # 连通率 = 匹配数 / 较大端点数（取两个方向的平均）
    cr_pred = matched_pred / len(pred_endpoints) if len(pred_endpoints) > 0 else 0
    cr_gt = matched_gt / len(gt_endpoints) if len(gt_endpoints) > 0 else 0

    return (cr_pred + cr_gt) / 2.0


def calculate_intersection_accuracy(pred_skel, gt_skel, tolerance=5):
    """
    交叉点检测准确率 (Intersection Accuracy, IA)：
    交叉点定义：邻域（3x3）中非零像素数 >= 3 的骨架像素。
    返回 (IA-Precision, IA-Recall)。
    """
    def find_intersections(skel):
        """提取骨架交叉点"""
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
        intersections = np.argwhere((skel > 0) & (neighbor_count >= 3))
        return intersections  # shape: (N, 2)

    # 需要做聚类，因为交叉点往往是多个像素的簇
    def cluster_points(points, merge_dist=8):
        """将距离过近的交叉点像素合并为一个代表点"""
        if len(points) == 0:
            return np.array([]).reshape(0, 2)
        clustered = []
        used = set()
        for i, p in enumerate(points):
            if i in used:
                continue
            cluster = [p]
            for j, q in enumerate(points):
                if j != i and j not in used:
                    if np.sqrt(np.sum((p - q) ** 2)) < merge_dist:
                        cluster.append(q)
                        used.add(j)
            used.add(i)
            clustered.append(np.mean(cluster, axis=0))
        return np.array(clustered)

    pred_ints = find_intersections(pred_skel)
    gt_ints = find_intersections(gt_skel)

    pred_clustered = cluster_points(pred_ints)
    gt_clustered = cluster_points(gt_ints)

    if len(pred_clustered) == 0 and len(gt_clustered) == 0:
        return 1.0, 1.0
    if len(pred_clustered) == 0:
        return 0.0, 0.0
    if len(gt_clustered) == 0:
        return 0.0, 0.0

    # 匈牙利匹配（简化版：贪心匹配）
    # 计算距离矩阵
    dist_matrix = np.zeros((len(pred_clustered), len(gt_clustered)))
    for i, p in enumerate(pred_clustered):
        for j, g in enumerate(gt_clustered):
            dist_matrix[i, j] = np.sqrt(np.sum((p - g) ** 2))

    # 贪心匹配
    matched = 0
    used_pred = set()
    used_gt = set()
    # 按距离排序
    indices = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)
    for idx in range(len(indices[0])):
        pi, gi = indices[0][idx], indices[1][idx]
        if pi in used_pred or gi in used_gt:
            continue
        if dist_matrix[pi, gi] <= tolerance:
            matched += 1
            used_pred.add(pi)
            used_gt.add(gi)

    ia_precision = matched / len(pred_clustered) if len(pred_clustered) > 0 else 0
    ia_recall = matched / len(gt_clustered) if len(gt_clustered) > 0 else 0

    return ia_precision, ia_recall


# ==========================================
# 2. 在整个测试集上运行评估 (Run Evaluation)
# ==========================================

def evaluate_testset(model_path, test_dir, device='cuda'):
    """
    在论文的 Test Set 上计算平均指标，并输出可以直接填入论文表格的数据
    """
    # 初始化模型与数据
    from HRNet import HarnessHRNetV2 # 请确保能正确import
    from Datasets import HarnessDataset
    
    model = HarnessHRNetV2(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 测试集绝不能加数据增强
    test_dataset = HarnessDataset(test_dir, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    metrics = {
        'clDice': [],
        'Precision (Tol=3)': [],
        'Recall (Tol=3)': [],
        'F1-Score (Tol=3)': [],
        'Breakage Rate': [],
        'Connectivity Rate': [],
        'IA-Precision': [],
        'IA-Recall': []
    }
    
    print(f"[*] 🚀 开始评估测试集，共 {len(test_dataset)} 张图像...")
    
    with torch.no_grad():
        for _, (imgs, targets) in enumerate(tqdm(test_loader)):
            imgs, targets = imgs.to(device), targets.to(device)
            
            # 1. 网络推理
            preds = model(imgs)
            
            # 2. 转换为 Numpy，并提取网络预测的单像素骨架
            pred_np = preds[0].cpu().numpy().squeeze()
            pred_skel = smooth_and_skeletonize(pred_np, threshold=0.5)
            
            # 3. 提取 GT 的单像素骨架
            # 因为 targets 是有一定宽度(3-5px)的 ribbon，需要对其进行骨架化作为真值骨架
            gt_np = targets[0].cpu().numpy().squeeze()
            gt_skel = skeletonize(gt_np > 0.5).astype(np.uint8)
            
            # 如果某张图连 GT 都没有（预防万一），跳过
            if np.sum(gt_skel) == 0:
                continue
                
            # 4. 计算指标
            cldice = calculate_cldice(pred_skel, gt_skel)
            precision, recall, f1 = calculate_relaxed_metrics(pred_skel, gt_skel, tolerance=3)
            br = calculate_breakage_rate(pred_skel, gt_skel, tolerance=3)
            cr = calculate_connectivity_rate(pred_skel, gt_skel, tolerance=5)
            ia_prec, ia_rec = calculate_intersection_accuracy(pred_skel, gt_skel, tolerance=5)

            metrics['clDice'].append(cldice)
            metrics['Precision (Tol=3)'].append(precision)
            metrics['Recall (Tol=3)'].append(recall)
            metrics['F1-Score (Tol=3)'].append(f1)
            metrics['Breakage Rate'].append(br)
            metrics['Connectivity Rate'].append(cr)
            metrics['IA-Precision'].append(ia_prec)
            metrics['IA-Recall'].append(ia_rec)
            
    # ==========================================
    # 3. 输出论文表格所需的数据
    # ==========================================
    n = len(metrics['clDice'])
    print("\n" + "="*50)
    print("论文测试集最终评估结果 (Test Set Results)")
    print("="*50)
    print(f"Total Test Images evaluated  : {n}")
    print(f"Mean clDice                  : {np.mean(metrics['clDice']):.4f} ± {np.std(metrics['clDice']):.4f}")
    print(f"Mean Relaxed Precision       : {np.mean(metrics['Precision (Tol=3)']):.4f} ± {np.std(metrics['Precision (Tol=3)']):.4f}")
    print(f"Mean Relaxed Recall          : {np.mean(metrics['Recall (Tol=3)']):.4f} ± {np.std(metrics['Recall (Tol=3)']):.4f}")
    print(f"Mean Relaxed F1-Score        : {np.mean(metrics['F1-Score (Tol=3)']):.4f} ± {np.std(metrics['F1-Score (Tol=3)']):.4f}")
    print("-" * 50)
    print(f"Mean Breakage Rate (BR)      : {np.mean(metrics['Breakage Rate']):.4f} ± {np.std(metrics['Breakage Rate']):.4f}")
    print(f"Mean Connectivity Rate (CR)  : {np.mean(metrics['Connectivity Rate']):.4f} ± {np.std(metrics['Connectivity Rate']):.4f}")
    print(f"Mean IA-Precision            : {np.mean(metrics['IA-Precision']):.4f} ± {np.std(metrics['IA-Precision']):.4f}")
    print(f"Mean IA-Recall               : {np.mean(metrics['IA-Recall']):.4f} ± {np.std(metrics['IA-Recall']):.4f}")
    print("="*50)

if __name__ == "__main__":
    # 请修改为您保存的最优模型路径和测试集路径
    MODEL_WEIGHTS = "./202605131048/best_model.pth"
    TEST_DATA_DIR = "./data/val"  # 可以分别测试 easy, medium, hard
    
    evaluate_testset(MODEL_WEIGHTS, TEST_DATA_DIR)