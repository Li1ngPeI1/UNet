# utils/other_utils.py

import matplotlib.pyplot as plt
import torch
from PIL import Image

def show_tensor_image(tensor):
    """
    将形状为 (C, H, W) 或 (B, C, H, W) 的 Tensor 显示为图像。
    若存在 batch 维度，则只显示第一张图。
    """
    # 去除 batch 维度（如果存在）
    if tensor.dim() == 4:
        tensor = tensor[0]
    # 转换为 (H, W) 或 (H, W, C)
    img = tensor.cpu().detach()
    if img.dim() == 3:
        img = img.permute(1, 2, 0) if img.shape[0] in [1, 3] else img
    # 若是单通道，squeeze 掉通道维
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    elif img.shape[0] == 1:
        img = img.squeeze(0)
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.axis('off')

def to_image(tensor):
    """
    将 Tensor 转换为 PIL Image 对象。
    支持单张图像 (C,H,W) 或网格图像 (C,H,W) 自动处理通道顺序。
    """
    img = tensor.cpu().detach()
    if img.dim() == 3:
        img = img.permute(1, 2, 0)  # CHW -> HWC
    if img.max() > 1.0:
        img = img / 255.0
    img = img.clamp(0, 1)
    img = (img * 255).byte().numpy()
    return Image.fromarray(img)