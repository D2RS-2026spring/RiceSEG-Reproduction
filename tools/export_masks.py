import argparse
import os
import cv2
import mmcv
import numpy as np
import torch
from mmseg.apis import init_model, inference_model
from mmengine import Config
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Export raw masks for downstream tasks')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--img-dir', help='Image directory path (Validation set)')
    parser.add_argument('--out-dir', help='Output directory for masks')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', action='store_true', help='Save with palette (color) or raw index (gray)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. 准备输出目录
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 2. 初始化模型
    model = init_model(args.config, args.checkpoint, device=args.device)

    # 3. 获取图片列表
    # 支持常见的图片格式
    img_names = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Found {len(img_names)} images. Starting inference...")

    for img_name in tqdm(img_names):
        img_path = os.path.join(args.img_dir, img_name)
        
        # 4. 推理
        result = inference_model(model, img_path)
        
        # 5. 提取 Mask
        # result 是 SegDataSample 对象
        # pred_sem_seg.data 是 (1, H, W) 的 Tensor
        mask_tensor = result.pred_sem_seg.data
        
        # 转为 numpy, 去掉 batch 维度 -> (H, W)
        # 此时 mask_np 里的值是类别索引：0, 1, 2...
        mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

        # 6. 保存
        # 注意：这里保存的是单通道灰度图。
        # 如果是二分类，像素值只有 0 和 1。肉眼看是全黑的，但这正是你的算法需要的。
        # 文件名保持一致，保存为 png 以免压缩损失
        save_name = os.path.splitext(img_name)[0] + '.png'
        save_path = os.path.join(args.out_dir, save_name)
        
        # 如果你确实需要看一眼 mask 是否正确，可以临时乘以 255 (仅用于调试，不要用于生产！)
        # cv2.imwrite(save_path, mask_np * 255) 
        
        # 生产环境直接保存原始值
        cv2.imwrite(save_path, mask_np)

    print(f"\nDone! Masks saved to: {args.out_dir}")
    print("注意：保存的图片像素值为 0,1,2... 肉眼查看可能为全黑，请使用代码读取验证。")

if __name__ == '__main__':
    main()