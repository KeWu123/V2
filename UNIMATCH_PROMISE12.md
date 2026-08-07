# PROMISE12 Baseline + UniMatch 实验说明

## 1. 实验边界

本实验不是加载已经训练好的 baseline 后继续 tuning，也不是调用另一个
Baseline 实验的 checkpoint。`train_unimatch.py` 会独立地从随机初始化开始，
先运行 1000 次监督预训练，再从本次预训练的最佳权重开始运行 5000 次
self-training。

原始 baseline 文件 `code/train_baseline.py` 未修改。两组实验保持相同的：

- PROMISE12 划分：42 个训练病例、4 个验证病例、4 个测试病例；
- 前 8 个训练病例为有标签集，共 242 个有标签切片；
- seed：1337；
- U-Net 结构与通道数；
- pretrain / self-train：1000 / 5000；
- batch size / labeled batch size：24 / 12；
- SGD、learning rate 0.01、momentum 0.9、weight decay 1e-4；
- 固定学习率；
- EMA decay 0.99；
- self-training 前 1000 次不使用无监督损失；
- baseline consistency 权重：
  按 `iteration / max_iterations` 归一化，并在 self-training 结束时完成 ramp-up；
- 每 200 次在同一个 validation split 上验证并保存最佳 Student；
- 监督损失为 `0.5 * (CE + Dice)`。

因此结果应与原 baseline 的独立同种子运行比较，不能与本方法自己的
pretrain 结果代替 baseline 比较。

## 2. 论文和官方实现中真正的 UniMatch

依据 CVPR 2023 论文 *Revisiting Weak-to-Strong Consistency in
Semi-Supervised Semantic Segmentation* 和作者的官方代码，UniMatch V1 包含
三个由同一个弱视图伪标签监督的分支：

1. 强图像视图 1，权重 0.25；
2. 强图像视图 2，权重 0.25；
3. 弱视图的特征扰动分支，权重 0.50。

官方医学图像版本使用 U-Net，对五级 encoder feature 全部应用
`Dropout2d(p=0.5)`，使用置信阈值 0.95，并在强视图上使用 ColorJitter、
Gaussian blur 和 CutMix。官方医学代码的无监督损失使用置信度掩码 Dice。

参考：

- 论文：<https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html>
- 官方仓库：<https://github.com/LiheYoung/UniMatch>
- 官方医学实现：<https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical>

## 3. 如何融合到当前 baseline

这里采用的是“保留 Mean Teacher baseline，替换无监督观察方式”的融合，
而不是把 baseline 改成官方纯 Student UniMatch：

```text
unlabeled weak image
        |
        +--> EMA Teacher (eval) --> softmax --> LCC hard pseudo label
        |                                  --> target-class confidence
        |
        +--> Student encoder --> feature Dropout --> decoder --> L_fp
        |
        +--> strong MRI view 1 + CutMix --> Student --> L_s1
        |
        +--> strong MRI view 2 + CutMix --> Student --> L_s2

L_uni = 0.25 L_s1 + 0.25 L_s2 + 0.50 L_fp
L_total = L_supervised + baseline_consistency_weight * L_uni
```

保留了 baseline 的以下原理：

- 伪标签仍由修正后的 EMA Teacher 生成，Teacher 固定在 eval 模式；
- Teacher 仍接收弱/原始视图；
- hard pseudo mask 仍保留每张 2D 切片的最大前景连通域；
- 每个分支内部仍使用 baseline 的 `0.5 * (CE + Dice)` 伪标签损失；
- EMA、ramp-up、1000 次截断和总损失外层尺度均未改变。

增加的仅是 UniMatch 的三项核心机制：双强视图、特征扰动、置信度筛选。

## 4. PROMISE12 必须做的域适配

PROMISE12 H5 图像在转换时做了前景 z-score 标准化，数值既不是 RGB，也不在
`[0, 1]`。官方 ACDC loader 中的 `uint8(img * 255)` 不能直接照搬，否则负值
和大于 1 的值会截断或回绕，得到的已经不是合理 MRI 强增强。

本实现的两路强视图保持空间坐标不变，在浮点域独立执行：

- 概率 0.8 的亮度与对比度扰动；
- 概率 0.5、sigma 0.1 到 2.0 的 Gaussian blur；
- 概率 0.5 的 CutMix，面积和长宽比范围与官方实现一致。

CutMix 同步混合图像、pseudo label 和 confidence，避免图像与监督错位。
两路 CutMix 的 donor permutation 与 box 独立采样。

需要明确：原 comparison baseline 仍然是“BCP 去掉 copy-paste”，没有被改动；
但官方 UniMatch 本身包含 CutMix，所以完整的 UniMatch 融合实验必然重新引入一种
区域复制混合。它不是 BCP 的 class-specific copy-paste，但同样属于区域混合增强。
默认保留 CutMix 是为了先做论文/官方代码一致的主实验；若设
`--cutmix_prob 0`，得到的是 no-CutMix 消融，不应再称为完整 UniMatch 主结果。

LCC 可能把 Teacher 原始 argmax 的小前景块改成背景。为避免这类像素仍以
“原前景 max probability”通过阈值，本实现按 LCC 后的 target class 从
Teacher softmax 中重新 gather confidence。

## 5. 为什么记录两种 coverage

前列腺只占每张切片很小一部分。只记录所有像素的置信 coverage 会被容易的
背景像素主导，即使 coverage 很高，也不代表前景伪标签可靠。因此日志同时输出：

- `coverage`：所有像素中达到 0.95 的比例；
- `fg_coverage`：Teacher 预测为前景的像素中达到 0.95 的比例。

判断 UniMatch 是否真的工作时，至少同时观察 validation Dice、`coverage`、
`fg_coverage`、`s1/s2/fp` 三个分支损失。若总 coverage 很高但 fg coverage
接近 0，说明阈值筛选几乎只在强化背景，这通常会导致前景召回下降。

## 6. RTX 5090 运行

在服务器的本文件夹中执行：

```bash
cd ~/zhengtaoma/Baseline
bash run_unimatch_5090.sh
```

默认实验名为 `MT_PROMISE12_UniMatch`。脚本会输出到：

```text
model/MT_PROMISE12_UniMatch_7_labeled/pre_train/unet/
model/MT_PROMISE12_UniMatch_7_labeled/self_train/unet/
```

后台运行：

```bash
DETACH=1 bash run_unimatch_5090.sh
```

若重复运行同一配置，不要覆盖旧结果，应只改实验名：

```bash
EXP_NAME=MT_PROMISE12_UniMatch_seed1337_run2 bash run_unimatch_5090.sh
```

训练完成后测试并直接生成指标表：

```bash
bash test_and_quantify_unimatch_5090.sh
```

自定义实验名训练时，测试必须传入同一个名字：

```bash
EXP_NAME=MT_PROMISE12_UniMatch_seed1337_run2 \
bash test_and_quantify_unimatch_5090.sh
```

最终会在对应实验目录生成 `metric_table.csv`、`metric_table.md` 和
`test_case_metrics.csv`。
