# RiceSEG Dataset Reproduction with SegFormer

本项目基于 MMSegmentation 框架，复现了《Global Rice Multi-Class Segmentation Dataset (RiceSEG)》论文中的基准实验，使用了 SegFormer (Mit-b0) 模型。

---

## 1. 环境准备

请参考 MMSegmentation 官方文档安装 PyTorch 和 MMCV，然后克隆本项目并编译：

```bash
git clone https://github.com/broucemonkey-stack/RiceSEG-Reproduction
cd mmsegmentation
pip install -v -e .
```

---

## 2. 数据准备

1. 请前往 [论文官方或数据集发布地址] 下载 RiceSEG 数据集。
2. 将数据集解压并组织成如下 MMSegmentation 标准格式，放入 `data/mmseg_format/` 目录下：

```text
data/
└── mmseg_format/
    ├── images/
    │   ├── train/
    │   └── val/
    └── masks/
        ├── train/
        └── val/
```

---

## 3. 训练模型命令

### 训练
python tools/train.py \\
configs/_custom/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \\
--work-dir work_dirs/RiceSEG/train

### tensorboard监测命令
tensorboard --logdir work_dirs/RiceSEG/train --port 6006

### 测试
python tools/test.py \\
work_dirs/RiceSEG/train/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \\
work_dirs/RiceSEG/train/iter_40000.pth \\
--show-dir work_dirs/RiceSEG/val \\
--work-dir work_dirs/RiceSEG/val

### 导出masks
python tools/export_masks.py \\
    work_dirs/RiceSEG/val/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \\
    work_dirs/RiceSEG/train/iter_40000.pth \\
    --img-dir data/mmseg_format/images/test \\
    --out-dir work_dirs/RiceSEG/test


---
