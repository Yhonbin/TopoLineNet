import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from HRNet import HarnessHRNetV2
from utils import predict_centerline

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用第一块GPU
OUTPUT_DIR = "./"

model = HarnessHRNetV2(model_name='hrnet_w18', pretrained=False)
static_dict = torch.load('./202605131048/best_model.pth',map_location='cuda')
model.load_state_dict(static_dict)
model.eval()
model.to('cuda')

# sample_img_path = './img0.jpg'
sample_img_path = './test_data/ins7.jpg'


if os.path.exists(sample_img_path):
        orig, heat, mask = predict_centerline(model, sample_img_path, 'cuda')
        if orig is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, "final_inference_mask.png"), mask)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "final_inference_heatmap.png"), heat)
            print(f"Inference test successful. Result saved in {OUTPUT_DIR}")