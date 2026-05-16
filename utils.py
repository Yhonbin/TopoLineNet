from skimage.morphology import skeletonize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def letterbox_image(image, expected_size):
    """
    等比例缩放并填充黑边 (Letterbox)
    彻底解决手机高分辨率图像直接 Resize 导致的长宽比破坏和旋转畸变问题。
    """
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(ew / iw, eh / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((eh, ew, 3), dtype=np.uint8)
    new_image[(eh - nh) // 2 : (eh - nh) // 2 + nh, 
              (ew - nw) // 2 : (ew - nw) // 2 + nw, :] = image
    
    # 返回缩放后的图，以及用于对齐热力图的偏移量信息
    return new_image, scale, (ew - nw) // 2, (eh - nh) // 2


# ==========================================
# 后处理优化技巧 (去毛刺 + 提取平滑骨架)
# 此部分供你在推理阶段或画图保存时使用
# ==========================================
def smooth_and_skeletonize(pred_mask_np, threshold=0.5):
    """
    解决骨架化出现“树枝分叉”的问题
    """
    # 1. 二值化
    binary_mask = (pred_mask_np > threshold).astype(np.uint8) * 255
    
    # 2. 闭运算 (Morphological Close)：填补线束内部可能断裂的小孔，平滑边界
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. 开运算 (Morphological Open)：去除边缘凸起的毛刺 (这些毛刺是产生树枝的罪魁祸首)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
    
    # 4. 骨架化提取中心线
    # skeletonize 需要布尔类型的输入 (True/False)
    skel = skeletonize(smoothed_mask > 0)
    
    return skel.astype(np.float32)


def predict_centerline(model, image_path, device, img_size=(512, 512), threshold=0.5):
    """
    进行推理并返回对齐原图尺寸的热力图与二值骨架 Mask
    """
    model.eval()
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not read image {image_path}")
        return None, None, None
        
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    
    # =========================================
    # 1. 预处理修复：使用 Letterbox 保持宽高比
    # =========================================
    # img_lb 是带黑边的 512x512 图像
    # scale 是缩放比例，pad_w 和 pad_h 是单侧填充的黑边宽度
    img_lb, scale, pad_w, pad_h = letterbox_image(img_rgb, img_size)
    
    img_tensor = torch.from_numpy(img_lb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        pred_heatmap = model(img_tensor)
    
    # 拿到 512x512 的网络输出
    heatmap_np = pred_heatmap.squeeze().cpu().numpy()
    
    # =========================================
    # 2. 后处理修复：逆向 Letterbox 映射
    # =========================================
    # 计算有效图像区域在 512x512 中的真实宽高
    nh = int(h_orig * scale)
    nw = int(w_orig * scale)
    
    # 关键步：把上下左右填充的黑边 (Padding) 裁掉，只保留有效的图像热力区域
    heatmap_cropped = heatmap_np[pad_h : pad_h + nh, pad_w : pad_w + nw]
    
    # 把裁剪后的有效热力图，放大回原始照片的真实尺寸
    heatmap_resized = cv2.resize(heatmap_cropped, (w_orig, h_orig))
    
    
    # =========================================
    # 3. 生成与训练时一致的伪彩色热力图 (重点修改部分)
    # =========================================
    # 将 [0.0, 1.0] 的浮点数映射到 [0, 255] 的 uint8 格式
    heatmap_8bit = np.clip(heatmap_resized * 255, 0, 255).astype(np.uint8)
    
    # 应用 JET 颜色映射 (深蓝->青->黄->红)，这就是你训练图里那种视觉效果
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    # =========================================
    # 4. 二值化 Mask
    # =========================================
    # 生成基础二值化图
    _, binary_mask = cv2.threshold((heatmap_resized * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # （可选）建议加上我们之前的形态学闭开运算，消除边缘毛刺，让 mask 更平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return img_rgb, heatmap_color, binary_mask,