import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
from pathlib import Path
from utils import letterbox_image

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[Warning] albumentations not installed. Using basic augmentation. Install: pip install albumentations")

# ==========================================
# Module 2: Data & Augmentation
# ==========================================

def build_augmentation_pipeline(p=0.7):
    """
    构建线束数据增强管道。

    几何变换（image + mask 同步）：
      - 随机旋转 90° 倍数
      - 水平/垂直翻转
      - 弹性变形（模拟线束弯曲）
      - 网格畸变
      - 仿射变换（缩放、平移、剪切）

    光照变换（仅 image）：
      - 亮度/对比度抖动
      - 高斯噪声
      - 高斯模糊
      - 色彩抖动

    遮挡模拟（仅 image）：
      - 随机擦除白色矩形（模拟标签/贴纸遮挡）
    """
    if not HAS_ALBUMENTATIONS:
        return None

    # ===== 几何变换：image 和 mask 必须同步 =====
    geometric = A.OneOf([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            shear=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8
        ),
    ], p=p)

    # ===== 光照变换：仅作用于 image =====
    photometric = A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=1.0
        ),
        A.GaussNoise(
            std_range=(0.01, 0.04),
            p=1.0
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
            p=1.0
        ),
    ], p=0.5)

    # ===== 遮挡模拟：随机白色矩形，模拟标签/贴纸 =====
    occlusion = A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(20, 80),
        hole_width_range=(20, 80),
        fill=230,   # 白色标签
        fill_mask=None,
        p=0.4
    )

    pipeline = A.Compose([geometric, photometric, occlusion])

    return pipeline


class HarnessDataset(Dataset):
    def __init__(self, data_dir, img_size=(512, 512), line_thickness=4, augment=False):
        self.data_dir = Path(data_dir)

        # 兼容你新建的 val/label 和 val/image 文件夹结构
        self.label_dir = self.data_dir / 'label' if (self.data_dir / 'label').exists() else self.data_dir
        self.image_dir = self.data_dir / 'image' if (self.data_dir / 'image').exists() else self.data_dir

        self.json_files = list(self.label_dir.glob("*.json"))
        self.img_size = img_size
        # 直接控制线宽，512分辨率下 3-5 个像素是比较合理的实体线宽
        self.line_thickness = line_thickness
        self.augment = augment
        # 初始化 albumentations 增强管道
        self.aug_pipeline = build_augmentation_pipeline(p=0.7) if augment else None  

    def __len__(self):
        return len(self.json_files)

    def _generate_heatmap(self, shapes, h, w, scale, pad_w, pad_h):
        # 1. 创建纯黑 Mask，直接作为语义分割的 Target
        mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.float32)
        
        for s in shapes:
            if s['shape_type'] in ['linestrip', 'polyline']:
                pts = np.array(s['points'], dtype=np.float32)
                pts[:, 0] = pts[:, 0] * scale + pad_w
                pts[:, 1] = pts[:, 1] * scale + pad_h
                pts = pts.astype(np.int32)
                
                # 绘制具有一定宽度的实心线 (Ribbon)
                cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=self.line_thickness)

        # 2. 加入极其轻微的高斯模糊，仅仅是为了抗锯齿(Anti-aliasing)，让边缘过渡平滑，而不是造成光晕
        if mask.max() > 0:
            mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.5)
            
        return mask

    def __getitem__(self, idx):
        j_path = self.json_files[idx]
        with open(j_path, 'r') as f: data = json.load(f)
        
        img_name = data.get('imagePath', j_path.stem + ".jpg")
        img_path = self.image_dir / img_name
        if not img_path.exists():
            for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg']:
                temp_path = j_path.with_suffix(ext)
                if temp_path.exists():
                    img_path = temp_path
                    break
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_lb, scale, pad_w, pad_h = letterbox_image(img, self.img_size)
        target_line = self._generate_heatmap(data['shapes'], img.shape[0], img.shape[1], scale, pad_w, pad_h)

        # 使用 albumentations 进行数据增强（image 和 mask 同步变换）
        if self.augment and self.aug_pipeline is not None:
            # albumentations 要求 image 为 uint8, mask 为 uint8 或 float32
            img_aug = img_lb.astype(np.uint8)
            mask_aug = target_line.astype(np.float32)
            result = self.aug_pipeline(image=img_aug, mask=mask_aug)
            img_lb = result['image']
            target_line = result['mask']

        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
        t_line_t = torch.from_numpy(target_line).unsqueeze(0).float()
        return img_t, t_line_t


class UnlabeledHarnessDataset(Dataset):
    """"无标签数据集"""
    def __init__(self, data_dir, img_size=(512, 512), augment=False):
        self.data_dir = Path(data_dir)
        # 支持多种常见图像格式
        self.img_files = list(self.data_dir.glob('*.jpg')) + \
                         list(self.data_dir.glob('*.png')) + \
                         list(self.data_dir.glob('*.jpeg'))

        # 过滤掉已经有同名 json 的图片，确保它们是纯无标签数据
        self.img_files = [f for f in self.img_files if not f.with_suffix('.json').exists()]
        self.img_size = img_size
        self.augment = augment
        # 无标签数据只需要光照增强（无 mask 同步需求）
        self.aug_pipeline = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
            ], p=0.5),
        ]) if augment and HAS_ALBUMENTATIONS else None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 无标签数据同样必须 Letterbox
        img_lb, _, _, _ = letterbox_image(img, self.img_size)

        # 光照增强（无标签数据不需要几何变换，因为 teacher 的 pseudo-label 也要同步变换）
        if self.augment and self.aug_pipeline is not None:
            img_lb = self.aug_pipeline(image=img_lb.astype(np.uint8))['image']

        img_t = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0

        return img_t
    
    

