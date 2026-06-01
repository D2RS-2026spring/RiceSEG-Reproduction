# 使用 SegFormer 复现 RiceSEG 数据集

本项目基于 MMSegmentation 框架，复现了《Global rice multiclass segmentation dataset (RiceSEG): comprehensive and diverse high-resolution RGB-annotated images for the development and benchmarking of rice segmentation algorithms》论文中的基准实验，使用 SegFormer (MiT-B0) 模型完成水稻多类别语义分割实验。

---

## 1. 结果复现

部分复现结果图已放置于 `assets/compare` 和 `assets/pred_only` 文件夹下。

**原图、模型分割图、人工标注图对比：**

<img src="./assets/compare/17_subset_overlap_0_1.png" alt="原图、模型分割图、人工标注图对比" width="100%">

图中依次展示了原始图像、模型预测结果和人工标注结果，可用于直观比较模型在水稻不同类别区域上的分割效果。

---

## 2. 环境准备

本项目基于 MMSegmentation 框架实现。请先参考 MMSegmentation 官方文档安装 PyTorch、MMCV 以及相关依赖，然后克隆本项目并进行环境配置。

```bash
git clone https://github.com/broucemonkey-stack/RiceSEG-Reproduction.git
cd RiceSEG-Reproduction
pip install -v -e .
```

如果本地已经安装好 MMSegmentation 环境，也可以根据实际环境路径进入对应项目目录后再执行相关训练和测试命令。

---

## 3. 数据准备

请前往 [RiceSEG 数据集官网](https://www.global-rice.com/) 下载 RiceSEG 数据集。

下载完成后，将数据集解压并整理为 MMSegmentation 支持的标准格式，建议放入 `data/mmseg_format/` 目录下，目录结构如下：

```text
data/
└── mmseg_format/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/
        ├── val/
        └── test/
```

其中，`images` 文件夹用于存放原始图像，`masks` 文件夹用于存放对应的语义分割标注图。训练、验证和测试数据应分别放入 `train`、`val` 和 `test` 子文件夹中。

---

## 4. 训练模型命令

### 4.1 训练模型

运行以下命令开始训练 SegFormer (MiT-B0) 模型：

```bash
python tools/train.py \
configs/_custom/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
--work-dir work_dirs/RiceSEG/train
```

训练过程中生成的日志文件、模型权重和中间结果会保存在 `work_dirs/RiceSEG/train` 目录下。

---

### 4.2 TensorBoard 监测

训练过程中可以使用 TensorBoard 查看损失变化、指标变化等训练信息：

```bash
tensorboard --logdir work_dirs/RiceSEG/train --port 6006
```

启动后，在浏览器中打开 TensorBoard 页面即可查看训练过程。

---

### 4.3 测试模型

训练完成后，可以使用以下命令在验证集或测试集上进行模型测试：

```bash
python tools/test.py \
work_dirs/RiceSEG/train/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
work_dirs/RiceSEG/train/iter_40000.pth \
--show-dir work_dirs/RiceSEG/val \
--work-dir work_dirs/RiceSEG/val
```

其中，`iter_40000.pth` 为训练得到的模型权重文件。实际使用时，可根据训练结果替换为对应的权重文件名称。

---

### 4.4 导出 Masks

如果需要导出模型预测得到的 mask 文件，可以运行以下命令：

```bash
python tools/export_masks.py \
work_dirs/RiceSEG/val/segformer_mit-b0_8xb2-160k_ade20k-512x512.py \
work_dirs/RiceSEG/train/iter_40000.pth \
--img-dir data/mmseg_format/images/test \
--out-dir work_dirs/RiceSEG/test
```

导出的预测结果会保存在 `work_dirs/RiceSEG/test` 目录下。

---

## 5. Mask 可视化

导出 mask 后，可以运行以下代码对预测结果进行可视化处理：

```bash
python code/visualize_masks.py
```

可视化结果可用于展示模型预测图、人工标注图以及原始图像之间的对比效果，便于分析模型在不同水稻类别上的分割表现。

---

## 6. 项目目录说明

本项目主要目录说明如下：

```text
RiceSEG-Reproduction/
├── assets/                 # 结果展示图片
│   ├── compare/             # 原图、预测图、标注图对比结果
│   └── pred_only/           # 单独的预测结果图
├── code/                   # 辅助处理和可视化脚本
├── configs/                # 模型配置文件
├── data/                   # 数据集目录
├── tools/                  # 训练、测试和导出相关脚本
├── work_dirs/              # 训练日志、模型权重和输出结果
└── README.md               # 项目说明文档
```

---

## 7. 注意事项

1. 运行训练和测试命令前，请确认数据集路径与配置文件中的路径保持一致。
2. 如果使用不同版本的 PyTorch、MMCV 或 MMSegmentation，可能需要根据本地环境调整配置。
3. 测试时需要保证模型权重文件路径正确，否则程序无法正常加载模型。
4. 如果需要更换测试图片目录，请修改 `--img-dir` 参数对应的路径。
5. 可视化脚本运行前，请确认预测 mask 文件已经成功导出。

---

## 8. 参考

* RiceSEG Dataset: https://www.global-rice.com/
* MMSegmentation: https://github.com/open-mmlab/mmsegmentation
* SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

