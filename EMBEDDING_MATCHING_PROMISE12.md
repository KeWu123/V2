# PROMISE12 Baseline + UniMatch + Embedding Matching v2

这一版是对失败版 Embedding Matching 的完整重实现。训练仍然从头执行
`pretrain 1000 + self-train 5000`，不是在旧 checkpoint 上继续 tuning。原始
Baseline 与 UniMatch 文件均未修改；实验仍由独立的
`code/train_embedding_matching.py` 运行。

## 保持不变的 Baseline/UniMatch

- PROMISE12，同一份 case-level 划分；
- 8 个 labeled cases、242 个 labeled slices；
- seed 1337；
- pretrain 1000 iterations，self-train 5000 iterations；
- batch size 24，labeled batch size 12；
- 原 U-Net、SGD、学习率、EMA 更新；
- self-train 前 1000 iterations 为 supervised-only；
- UniMatch 的两个 strong views、CutMix、feature dropout；
- 原 `EMA + 2D LCC + confidence >= 0.95` hard pseudo target；
- 原 UniMatch 三分支损失及权重 `(0.25, 0.25, 0.50)`。

## v2 为什么重写

旧版并不是论文中的 Embedding Matching：它把 EMA Teacher 特征同时用于
labeled reference 与 unlabeled query，使用跨 iteration FIFO、四区域 top-k
最大相似度和多重接受阈值，并把匹配损失重复加到三个 UniMatch 分支。实际日志中
最终只接受约 `0.06%` 像素、每个 batch 约 6 个，而且匹配前景比例长期为 0；
少量背景目标的归一化损失却很大，因此结果从 UniMatch 的约 0.844 降到了约
0.826。

v2 删除了 FIFO、top-k 最大值、相似度/置信度/margin 二次筛选以及三分支重复
监督，恢复论文的 `L_NN + L_EN` 主体。

## v2 的计算流程

1. 从当前 batch 的真实 labeled mask 提取目标内侧和背景外侧的 2D 表面带。
2. 使用 **EMA Teacher 的 labeled decoder features** 作为参考特征。
3. 使用 **Student 的 unlabeled decoder features** 作为查询特征。
4. 每个类别随机采样 `k=16` 个归一化表面特征，计算平均 cosine similarity；
   独立采样 `l=5` 次并平均五个 dense similarity maps。
5. 对前景/背景 similarity maps 做 softmax；较大者产生 detached hard NN target。
6. EMA Teacher 进行 `M=5` 次 MC-Dropout，并加入标准差 0.01、截断 0.02 的
   weak Gaussian noise。预测均值的 entropy 用于识别不可靠像素：

   `lambda(T) = [0.75 + 0.25 * sigmoid_rampup(T, T_N)] * ln(2)`。

7. Embedding Matching 只处理同时满足以下条件的像素：
   - MC predictive entropy 不低于 `lambda(T)`；
   - 原 UniMatch confidence 小于 0.95，即原分支本来会忽略的像素。
8. `L_NN` 用 hard NN target 监督一次 regular Student logits；不会再复制到两个
   strong 分支和 feature-dropout 分支。
9. `L_EN` 最小化 NN classifier entropy，梯度保留到 Student unlabeled feature，
   用于真正分开前景/背景 embedding。hard target 本身按论文要求 detach。
10. 总损失为：

    `L = L_sup + w_uni * L_uni + w_em * (L_NN + L_EN)`。

`w_em` 按论文 Gaussian ramp-up 到 0.125。PROMISE12 是小目标 2D slice，而原论文
是 3D hip CT patch；为避免再次出现“极少像素却按像素数归一化后主导梯度”，默认
按 5% active coverage 对 `w_em` 做下限比例缩放。这是明确的 2D 稳定性适配，不是
额外的伪标签接受阈值。训练入口保留了 `--em_coverage_reference` 参数，设为 0 即可
关闭；默认主实验先使用 0.05 的稳定配置。

## 开源代码与论文依据

主论文没有提供可确认的官方完整代码仓库，因此主体严格按论文公式实现，并对其
公开子模块使用官方开源代码：

- [Embedding Matching 论文（MIDL 2024）](https://openreview.net/pdf?id=xkqLQoFQbl)：Teacher/Student 特征方向、表面内外采样、`k=16`、`l=5`、平均 cosine、`L_NN`、`L_EN`、`w_PS=0.125`；
- [UA-MT 官方代码](https://github.com/yulequan/UA-MT/blob/master/code/train_LA_meanteacher_certainty_unlabel.py)：MC-Dropout、Gaussian input noise、平均概率的 predictive entropy 与动态 entropy threshold；
- [PPC 官方代码](https://github.com/IsYuchenYuan/PPC)：按类别处理像素表示与 prototype 的工程结构；其 [`unet_proto.py`](https://github.com/IsYuchenYuan/PPC/blob/main/code/networks/unet_proto.py) 中对 embedding 使用 `F.normalize` 后再计算类别相似度；
- [UniMatch 官方代码](https://github.com/LiheYoung/UniMatch)：现有两个 strong views、CutMix 和 feature perturbation 的依据。

这里没有把整套 UA-MT 或 PPC 拼入模型；只复用了 Embedding Matching 本身明确依赖
的 MC uncertainty 和规范化类别特征处理。

## RTX 5090 训练

```bash
cd ~/zhengtaoma/Baseline
CONDA_ENV=few_diffusion bash run_embedding_matching_5090.sh
```

后台运行：

```bash
CONDA_ENV=few_diffusion DETACH=1 bash run_embedding_matching_5090.sh
```

默认输出到新目录，避免与失败版混用：

```text
model/MT_PROMISE12_UniMatch_EmbeddingMatching_v2_7_labeled/
```

## 测试与量化

```bash
CONDA_ENV=few_diffusion bash test_and_quantify_embedding_matching_5090.sh
```

生成：

```text
metric_table.csv
metric_table.md
test_case_metrics.csv
pre_train/unet/performance.txt
self_train/unet/performance.txt
```

训练日志需要重点查看：

- `LNN`、`LEN`：两个新增损失；
- `em_w=raw/effective`：论文 ramp 权重与 coverage 适配后的实际权重；
- `uncertain`：高 MC uncertainty 像素比例；
- `active`：高 uncertainty 且被 UniMatch 忽略的实际匹配比例；
- `nn_fg`：active 像素中 NN 判为前景的比例，不应再长期为 0；
- `disagree`：NN target 与原 Teacher target 的差异比例；
- `nn_entropy`：匹配分类器 entropy，训练后应有下降趋势；
- `refs=(foreground, background)`：当前 batch 的真实表面参考数量。
