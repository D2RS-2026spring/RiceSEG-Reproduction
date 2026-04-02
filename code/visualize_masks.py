import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D

# 1. 定义类别到颜色和名称的映射
CLASS_INFO = {
    0: {'name': 'background',    'color': [0, 0, 0]},       
    1: {'name': 'green_veg',     'color': [0, 176, 80]},    
    2: {'name': 'senescent_veg', 'color': [255, 255, 0]},   
    3: {'name': 'panicle',       'color': [255, 0, 0]},     
    4: {'name': 'weed',          'color': [0, 112, 192]},   
    5: {'name': 'duckweed',      'color': [112, 48, 160]}   
}

COLOR_MAP = {k: v['color'] for k, v in CLASS_INFO.items()}

def apply_mask_overlay(image_np, mask_np, alpha=0.5):
    """
    将分割 Mask 叠加到原图上
    """
    overlay_color = np.zeros_like(image_np, dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        if class_id == 0:
            continue
        overlay_color[mask_np == class_id] = color
        
    active_mask = mask_np != 0
    blended_image = image_np.copy()
    blended_image[active_mask] = (
        image_np[active_mask] * (1 - alpha) + overlay_color[active_mask] * alpha
    ).astype(np.uint8)
    
    return blended_image


if __name__ == "__main__":
    # --- 文件夹路径配置 ---
    rgb_folder = r'E:\Open_dataset\RiceSEG_dataset\mmseg_format\images\test'
    pred_mask_folder = r'E:\Open_dataset\RiceSEG_dataset\mmseg_format\test'       # 模型分割mask
    gt_mask_folder = r'E:\Open_dataset\RiceSEG_dataset\mmseg_format\masks\test'   # 人工标注真值mask
    output_folder = r'E:\Open_dataset\RiceSEG_dataset\mmseg_format\result'        # 输出结果路径
    
    # 如果输出文件夹不存在，自动创建它
    os.makedirs(output_folder, exist_ok=True)

    # 1. 查找所有图片文件
    valid_exts = ('.jpg', '.png', '.jpeg')
    rgb_files = [f for f in os.listdir(rgb_folder) if f.lower().endswith(valid_exts)]
    
    print(f"✅ 找到 {len(rgb_files)} 张图片，开始批量生成...")

    # 2. 创建图例元素
    legend_elements = []
    for class_id, info in CLASS_INFO.items():
        if class_id == 0:
            markerfacecolor = 'white'
            markeredgecolor = '#0070C0' 
        else:
            markerfacecolor = np.array(info['color']) / 255.0
            markeredgecolor = markerfacecolor
            
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=info['name'],
                   markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor, 
                   markersize=14)
        )

    # 3. 遍历并处理每一张图片
    for i, rgb_name in enumerate(rgb_files):
        name_base = os.path.splitext(rgb_name)[0]
        mask_name = name_base + '.png'  

        # 组合各文件路径
        rgb_path = os.path.join(rgb_folder, rgb_name)
        pred_path = os.path.join(pred_mask_folder, mask_name)
        gt_path = os.path.join(gt_mask_folder, mask_name)
        
        # 定义两种输出结果的路径
        save_path_compare = os.path.join(output_folder, 'compare',f"{name_base}.png")
        save_path_pred_only = os.path.join(output_folder, 'pred_only', f"{name_base}.png")

        # 检查两个 Mask 是否都存在
        if not os.path.exists(pred_path):
            print(f"⚠️ [{i+1}/{len(rgb_files)}] 缺失模型预测 Mask，跳过: {mask_name}")
            continue
        if not os.path.exists(gt_path):
            print(f"⚠️ [{i+1}/{len(rgb_files)}] 缺失人工真值 Mask，跳过: {mask_name}")
            continue

        # 读取图像并强制格式化以防报错
        image_np = np.array(Image.open(rgb_path).convert("RGB"))
        pred_mask_np = np.array(Image.open(pred_path).convert("L"))
        gt_mask_np = np.array(Image.open(gt_path).convert("L"))

        # 生成叠加效果图 (设置 alpha 透明度，0.6 效果较好)
        blended_pred = apply_mask_overlay(image_np, pred_mask_np, alpha=0.6)
        blended_gt = apply_mask_overlay(image_np, gt_mask_np, alpha=0.6)
        
        # ==========================================
        # 任务一：生成并保存 1x3 三者对比图
        # ==========================================
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'wspace': 0, 'hspace': 0})
        
        axes1[0].imshow(image_np)
        axes1[0].axis('off')
        axes1[0].set_title('Original Image', pad=10, fontsize=14)
        
        axes1[1].imshow(blended_pred)
        axes1[1].axis('off')
        axes1[1].set_title('Model Prediction', pad=10, fontsize=14)
        
        axes1[2].imshow(blended_gt)
        axes1[2].axis('off')
        axes1[2].set_title('Ground Truth', pad=10, fontsize=14)

        # 添加图例到底部中央，并调整边距
        fig1.legend(handles=legend_elements, loc='lower center', ncol=6, 
                   bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=14)
        fig1.subplots_adjust(bottom=0.15)

        fig1.savefig(save_path_compare, dpi=300, bbox_inches='tight')
        plt.close(fig1) # 释放内存
        
        # ==========================================
        # 任务二：生成并保存只有【复现模型分割图】的单图
        # ==========================================
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        ax2.imshow(blended_pred)
        ax2.axis('off')
        
        # 添加相同的图例
        fig2.legend(handles=legend_elements, loc='lower center', ncol=3, 
                   bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
        fig2.subplots_adjust(bottom=0.15)
        
        fig2.savefig(save_path_pred_only, dpi=300, bbox_inches='tight')
        plt.close(fig2) # 释放内存

        print(f"💾 [{i+1}/{len(rgb_files)}] 已完成: {name_base}")

    print(f"\n🎉 全部处理完成！对比图和单张效果图已保存至: {output_folder}")