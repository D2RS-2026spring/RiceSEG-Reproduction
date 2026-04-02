# RiceSEG Dataset Reproduction with SegFormer

本项目基于 MMSegmentation 框架，复现了《Global Rice Multi-Class Segmentation Dataset (RiceSEG)》论文中的基准实验，使用了 SegFormer (Mit-b0) 模型。

## 1. 环境准备
请参考 MMSegmentation 官方文档安装 PyTorch 和 MMCV，然后克隆本项目并编译：
\`\`\`bash
git clone <你的GitHub仓库地址>
cd mmsegmentation
pip install -v -e .
\`\`\`

## 2. 数据准备
1. 请前往 [论文官方或数据集发布地址] 下载 RiceSEG 数据集。
2. 将数据集解压并组织成如下 MMSegmentation 标准格式，放入 `data/mmseg_format/` 目录下：
data/
└── mmseg_format/
    ├── images/
    │   ├── train/
    │   └── val/
    └── masks/
        ├── train/
        └── val/

## 3. 训练模型
运行以下命令开始训练（基于单卡 RTX 4080 配置优化）：
\`\`\`bash
python tools/train.py configs/segformer/你的配置文件名.py
\`\`\`

## 4. 测试与可视化
\`\`\`bash
python tools/test.py configs/segformer/你的配置文件名.py work_dirs/你的模型权重.pth --show-dir work_dirs/val_results
\`\`\`