# 农田秸秆覆盖率预测

本项目使用图像分割模型识别农田图片中的秸秆像素，并计算秸秆覆盖率：

```text
coverage = 预测为秸秆的像素数 / 图片总像素数
```

当前推荐模型是以 ImageNet 预训练 ResNet-34 为编码器的 SMP U-Net。项目同时保留了
Mini U-Net 和 Random Forest 两个历史模型，用于统一测试与效果比较。

## 项目结构

```text
residue_coverage/
├── predict.py
├── model.py
├── requirements.txt
├── MODEL_CHECKSUMS.sha256
├── smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth
├── smp_unet_resnet34_retrained_seed42.pth
├── smp_unet_resnet34_cpu_run_seed42.pth
│
├── training/
│   ├── main.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── evaluate_all_models.py
│   ├── test.py
│   ├── utils.py
│   └── visualiseE.py
│
├── legacy/
│   ├── classical_ml/
│   │   ├── residue_rf_model.joblib
│   │   ├── training.py
│   │   ├── test.py
│   │   └── visualiseE.py
│   └── unet/
│       ├── mini_unet_abc_bce+dice_seed42_train_generator.pth
│       ├── main.py
│       ├── test.py
│       └── visualiseE.py
│
├── residue_background/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── D/
│   └── E/
│
└── logs/
    ├── model_test_report.txt
    ├── model_test_metrics.csv
    ├── model_test_per_image.csv
    ├── smp_unet_mps_retraining.log
    ├── retrained_smp/
    ├── Figure_RF_MAE.png
    ├── Figure_MiniUNet_MAE.png
    └── Figure_smpUnet_MAE.png
```

主要文件说明：

- `predict.py`：使用当前 SMP U-Net 预测单张图片。
- `model.py`：当前 SMP U-Net 模型结构。
- `training/`：当前模型的训练、测试和统一评估代码。
- `legacy/classical_ml/`：Random Forest 模型及其历史代码。
- `legacy/unet/`：Mini U-Net 模型及其历史代码。
- `logs/`：测试指标、逐图结果、训练日志和报告图。
- `MODEL_CHECKSUMS.sha256`：模型文件的 SHA-256 校验值。

`smp_unet_resnet34_cpu_run_seed42.pth` 是保留用于对照的 CPU 训练版本，不是
`predict.py` 的默认模型。由于 `.pth` 文件体积较大并被 Git 忽略，复制或备份模型后建议
使用校验文件确认内容没有变化。

## 环境安装

推荐使用项目自带的虚拟环境；如果需要重新创建：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

以后进入项目时只需：

```bash
source .venv/bin/activate
```

验证当前模型文件是否完整：

```bash
shasum -a 256 -c MODEL_CHECKSUMS.sha256
```

## 命令行预测单张图片

最简单的用法是把图片地址作为第一个参数：

```bash
python predict.py /path/to/image.jpg
```

例如：

```bash
python predict.py residue_background/E/IMG_0941_part09.jpg
```

程序会：

1. 加载根目录中的默认 SMP U-Net checkpoint；
2. 对输入图片进行 ImageNet normalization；
3. 预测二值秸秆 mask；
4. 计算秸秆像素占整张图片像素的百分比；
5. 在命令行打印 coverage；
6. 在原图旁边保存一张 Matplotlib 结果图。

默认结果保存在输入图片旁边，文件名为：

```text
<原文件名>_coverage.png
```

结果图左侧为原图，右侧为预测 mask；mask 中白色表示预测的秸秆像素，标题中会显示
coverage 百分比。

### 指定输出位置

```bash
python predict.py image.jpg --output results/image_prediction.png
```

也可以使用简写：

```bash
python predict.py image.jpg -o results/image_prediction.png
```

### 显示 Matplotlib 窗口

结果默认只保存到文件。如果还需要打开窗口：

```bash
python predict.py image.jpg --show
```

### 指定模型、阈值或设备

```bash
python predict.py image.jpg \
  --checkpoint smp_unet_resnet34_retrained_seed42.pth \
  --threshold 0.5 \
  --device mps
```

设备选项：

- `auto`：优先 CUDA，其次 MPS，最后 CPU；
- `cuda`：NVIDIA GPU；
- `mps`：Apple Silicon GPU；
- `cpu`：CPU。

完整参数：

```bash
python predict.py --help
```

模型可以处理非 32 倍数尺寸的图片：脚本会自动补边到模型所需尺寸，预测后再裁回原图尺寸。

## 数据划分

所有图片都使用同名的 JPG/TIF 文件配对：

```text
IMG_xxxx_partxx.jpg
IMG_xxxx_partxx.tif
```

数据按拍摄位置划分，没有把不同位置随机混合：

| 用途 | 位置 | 图片数量 |
|---|---|---:|
| 训练 | A、B、C | 400 |
| 验证与 early stopping | D | 144 |
| 最终测试 | E | 144 |

二值 mask 的定义：

- `1`：秸秆；
- `0`：背景。

## 三个模型的 E 集测试效果

三个模型使用相同的 `residue_background/E` 测试集，共 144 张图片。

- Random Forest 阈值：`0.60`
- Mini U-Net 阈值：`0.50`
- SMP U-Net 阈值：`0.50`

### 分割指标

| 模型 | Pixel IoU | Mean IoU | Global Dice | Mean Dice | Precision | Recall | Specificity | Pixel accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6458 | 0.6458 | 0.7848 | 0.7818 | 0.9572 | 0.6650 | 0.8353 | 0.6911 |
| Mini U-Net | 0.7780 | 0.7755 | 0.8751 | 0.8714 | 0.9297 | 0.8266 | 0.6537 | 0.8002 |
| **SMP U-Net (ResNet-34)** | **0.8260** | **0.8241** | **0.9047** | **0.9021** | **0.9051** | **0.9043** | 0.4750 | **0.8387** |

`Pixel IoU` 和 `Global Dice` 先汇总所有测试像素再计算；`Mean IoU` 和 `Mean Dice`
是 144 张图片逐图计算后的平均值。

### Coverage 指标

测试集的真实总体 coverage 为 `84.70%`。

| 模型 | 预测 coverage | Coverage MAE | Coverage RMSE | Coverage bias | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 58.85% | 25.86% | 27.78% | -25.86% | 52.49% |
| Mini U-Net | 75.31% | 11.73% | 14.31% | -9.39% | 37.46% |
| **SMP U-Net (ResNet-34)** | **84.63%** | **7.71%** | **9.73%** | **-0.07%** | **27.31%** |

其中：

- `Coverage MAE`：每张图片 coverage 绝对误差的平均值；
- `Coverage RMSE`：对较大误差更敏感；
- `Coverage bias`：预测 coverage 减去真实 coverage，正值表示总体高估，负值表示总体低估。

### 结果解读

- Random Forest precision 很高，但 recall 较低，整体明显低估秸秆覆盖率。
- Mini U-Net 的分割和 coverage 结果明显优于 Random Forest，但仍存在约 `9.39`
  个百分点的总体低估。
- SMP U-Net 的 IoU、Dice 和 coverage MAE 都是三个模型中最佳；其总体 coverage
  与真实值只相差约 `0.07` 个百分点。

完整测试数据：

- `logs/model_test_report.txt`：便于阅读的完整汇总；
- `logs/model_test_metrics.csv`：三个模型的汇总指标；
- `logs/model_test_metrics.json`：JSON 格式汇总；
- `logs/model_test_per_image.csv`：三个模型共 432 条逐图指标。

报告图：

- `logs/Figure_RF_MAE.png`
- `logs/Figure_MiniUNet_MAE.png`
- `logs/Figure_smpUnet_MAE.png`

每张报告图包含三行：

1. coverage 绝对误差最低的图片；
2. coverage 绝对误差最接近该模型平均 MAE 的图片；
3. coverage 绝对误差最高的图片。

每一行依次展示原图、真实 mask、预测 mask 和 error map。Error map 中：

- 红色：false positive；
- 蓝色：false negative；
- 黑色：预测正确。

## 重新生成三模型测试报告

从项目根目录运行：

```bash
python -m training.evaluate_all_models
```

只测试 SMP U-Net：

```bash
python -m training.evaluate_all_models \
  --models smp_unet_resnet34
```

测试指定的 SMP checkpoint：

```bash
python -m training.evaluate_all_models \
  --models smp_unet_resnet34 \
  --smp-checkpoint smp_unet_resnet34_retrained_seed42.pth \
  --output-dir logs/retrained_smp
```

Apple Silicon 可以显式使用 MPS：

```bash
python -m training.evaluate_all_models --device mps
```

## 重新训练当前 SMP U-Net

当前训练配置：

| 配置 | 值 |
|---|---|
| 模型 | SMP U-Net |
| Encoder | ResNet-34 |
| Encoder 初始化 | ImageNet |
| 训练集 | A、B、C |
| 验证集 | D |
| Loss | BCE + soft Dice |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Batch size | 4 |
| Seed | 42 |
| 最大 epochs | 50 |
| Early-stopping patience | 10 |
| 选模指标 | 验证集 IoU |

在 Apple Silicon 上训练：

```bash
python -m training.main \
  --device mps \
  --output-checkpoint smp_unet_resnet34_training_candidate_seed42.pth
```

训练脚本拒绝覆盖已有 checkpoint。输出文件已经存在时，需要选择一个新的文件名，而不是
直接覆盖生产模型。

最近一次 MPS 重训在 Epoch 33 early stop，最佳 checkpoint 来自 Epoch 23：

| 数据集 | IoU | Dice | Precision | Recall | Coverage MAE |
|---|---:|---:|---:|---:|---:|
| 训练 A/B/C | 0.7778 | 0.8750 | 0.8942 | 0.8566 | 1.37% |
| 验证 D | 0.5538 | 0.7128 | 0.6593 | 0.7758 | 5.66% |

完整训练日志位于：

```text
logs/smp_unet_mps_retraining.log
```

## 历史模型命令

### Mini U-Net

```bash
python -m legacy.unet.main
python -m legacy.unet.test
python -m legacy.unet.visualiseE
```

### Random Forest

训练：

```bash
python -m legacy.classical_ml.training
```

测试 E 集：

```bash
python -m legacy.classical_ml.test
```

生成历史可视化：

```bash
python -m legacy.classical_ml.visualiseE
```

Random Forest 默认使用 `0.60` 概率阈值，可通过 `--threshold` 修改。

## 当前推荐

实际预测优先使用根目录的 SMP U-Net：

```text
smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth
```

它与最近重训输出的以下文件内容完全一致：

```text
smp_unet_resnet34_retrained_seed42.pth
```

推荐直接使用：

```bash
python predict.py /path/to/image.jpg
```
