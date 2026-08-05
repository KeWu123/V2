# UtilityMatch 模型思路与创新点说明

> 当前状态：方法与代码已经完成，完整 seed 1337 训练尚未产生最终测试结论。
> 本文只描述方法设计与相对 UniMatch 的真实变化，不把预期效果写成已经验证的结论。

## 1. 一句话概括

UtilityMatch 在不改变 U-Net、EMA Teacher、伪标签生成方式和 UniMatch
损失函数的前提下，把原来**随机抽取两路强增强**改为：先从完全相同的
UniMatch 增强分布中采样四路候选，再利用当前有标签数据提供的“干净任务梯度”
衡量每个候选的优化效用，最终只用效用最高的两路完成本次参数更新。

因此，UtilityMatch 的核心不是“增加一种图像变换”，而是提出一种：

> **由有标签任务梯度指导的、针对实际采样强增强视图的在线选择机制。**

它属于数据增强/伪监督优化方向，而不是新的分割网络，也不是在已训练好的
UniMatch 权重之后追加一个 refinement 阶段。

## 2. 原始 UniMatch 基线

当前 PROMISE12 UniMatch 自训练阶段包含三条无监督分支：

1. EMA Teacher 保持为有效本地实验所用的 `train()`，在未标注弱视图上生成 hard pseudo
   label，并保留每张切片的最大前景连通域；
2. 从同一增强分布中独立随机采样两路强视图，包括浮点域亮度/对比度、
   Gaussian blur 和 CutMix；
3. Student 还有一路 feature-dropout 扰动分支。

CutMix 同步作用于强图像、伪标签和置信度图。只有 Teacher 最大类别概率不低于
`tau=0.95` 的像素参与伪监督。设监督损失为：

```math
L_s = \frac{1}{2}\left(L_{CE}^{l}+L_{Dice}^{l}\right),
```

两路强视图损失和特征扰动损失分别为
`L_{u,s1}`、`L_{u,s2}` 和 `L_{u,fp}`，则原 UniMatch 的无监督目标为：

```math
L_u=0.25L_{u,s1}+0.25L_{u,s2}+0.50L_{u,fp},
```

总目标为：

```math
L=L_s+\lambda(t)L_u.
```

原方法的问题不在于没有强增强，而在于两路强增强是**盲目随机采样**的。
置信度阈值只能判断某个伪标签像素是否足够可信，却不能回答一个不同的问题：

> 当前实际采样到的强增强，产生的训练方向是否有利于真实分割任务？

## 3. 科学问题与核心假设

前序实验已经表明：强弱视图之间的高 JS、不一致或预测波动并不等价于有害噪声；
它们也可能代表模型需要学习的有效困难样本。直接过滤高风险视图可能同时删除有用
的训练信号。

UtilityMatch 因此不再尝试用 confidence、entropy、JS 或时间稳定性推测一张强视图
“看起来是否安全”，而是检验它对当前分割目标产生的优化方向：

> 如果某一路伪监督所产生的梯度，与当前真实标注监督损失的下降方向一致，那么这路
> 强增强更可能提供任务相关的有效学习信号；反之，它可能推动模型偏离有标签任务。

对应的可检验假设为：

> 在相同的 UniMatch 自训练更新次数下，使用 clean labeled-gradient utility 选择两路
> 实际强增强，比随机选择两路强增强获得更好的分割性能，并减少病例级严重退化。

## 4. UtilityMatch 的完整方法

### 4.1 构造候选强增强

完成前 1,000 iteration 的原有监督 warm-up 后，对每个未标注 batch 从原 UniMatch
增强分布独立采样 `K=4` 路候选：

```math
\tilde{x}_u^{(k)}=T_k(x_u),\qquad k=1,\ldots,4.
```

每一路候选都有独立的亮度/对比度和模糊参数、CutMix donor permutation 以及
CutMix rectangle。弱视图 EMA 产生的 pseudo label 与 confidence map 使用同一
permutation 和 rectangle 同步传输，避免图像与监督目标错位。

### 4.2 提取有标签任务参考方向

使用当前 batch 的有标签 CE+Dice 损失 `L_s`，只在 U-Net 最后的输出卷积
`decoder.out_conv` 上计算参考梯度：

```math
g_L=\nabla_{\theta_h}L_s,
```

其中 `theta_h` 表示最终分割输出层参数。该梯度来自真实标签，因此用于表示当前
mini-batch 的干净任务优化方向。

只选输出层而不是整个 U-Net，目的是降低四路候选在线评分的显存和计算成本，并让
选择指标直接对应最终类别决策空间。

### 4.3 计算候选伪监督梯度

对第 `k` 路候选，仍使用原 UniMatch 的 `tau=0.95` 置信度掩码 CE+Dice 伪监督损失：

```math
L_u^{(k)}=\frac{1}{2}
\left(L_{CE,mask}^{(k)}+L_{Dice,mask}^{(k)}\right),
```

并在同一个输出层上得到：

```math
g_k=\nabla_{\theta_h}L_u^{(k)}.
```

候选评分时，backbone/decoder feature 被 detach，只重新经过最终输出卷积，因此不保留
四套完整反向图。候选 forward 暂时关闭 BatchNorm running-buffer 更新，但模型仍处于
train mode；评分使用 `autograd.grad`，不会写入参数的 `.grad`，也不会直接更新模型。

### 4.4 定义 clean-gradient utility

第 `k` 路候选的效用定义为其伪监督梯度在干净监督梯度方向上的有符号投影：

```math
U_k=\frac{\langle g_k,g_L\rangle}
{\lVert g_L\rVert_2+\epsilon}.
```

这里使用的是**有符号投影而不是余弦相似度**：

- `U_k > 0`：候选伪监督梯度与有标签任务梯度总体同向；
- `U_k < 0`：二者存在方向冲突；
- 更大的正值同时保留方向一致性和候选梯度有效幅度的信息。

当前实现没有额外阈值、温度系数、学习型打分器或手工融合权重。

### 4.5 选择并正常训练

选择效用最大的两路：

```math
\mathcal{S}=\operatorname{Top2}_{k\in\{1,\ldots,4\}}U_k.
```

随后将这两路图像重新输入正常 train-mode Student，建立完整计算图，并继续使用原
UniMatch 损失：

```math
L_u^{UM}=0.25L_u^{(s_1)}+0.25L_u^{(s_2)}+0.50L_{u,fp}.
```

最终仍然执行一次正常的：

```math
L=L_s+\lambda(t)L_u^{UM}
```

反向传播、SGD 更新和 EMA Teacher 更新。Utility score 只决定本轮训练使用哪两路
强视图，不作为新的损失项，也不改变各分支权重。

## 5. 相对原 UniMatch 的真实变化

| 模块 | 原 UniMatch | UtilityMatch | 是否改变 |
|---|---|---|---|
| 分割网络 | 二维 U-Net | 同一二维 U-Net | 否 |
| 初始化 | 固定 Pre10000 最优权重及 SGD 状态 | 完全相同 | 否 |
| EMA Teacher | decay 0.99、固定 `train()` | 完全相同 | 否 |
| pseudo label | 弱视图 EMA、hard mask、最大连通域 | 完全相同 | 否 |
| 置信度筛选 | `tau=0.95` | 完全相同 | 否 |
| 强增强算子 | 亮度/对比度、blur、CutMix | 完全相同 | 否 |
| 强视图产生 | 随机采样 2 路并直接训练 | 随机采样 4 路，按 utility 选 2 路 | **是** |
| feature perturbation | dropout 0.5 | 完全相同 | 否 |
| 分支权重 | 0.25/0.25/0.50 | 完全相同 | 否 |
| 总损失与 ramp-up | 原 UniMatch 设置 | 完全相同 | 否 |
| optimizer/LR | SGD、当前 polynomial schedule | 完全相同 | 否 |
| 自训练更新数 | 30,000 | 30,000 | 否 |
| 验证模型 | online Student | online Student | 否 |
| 新增信息 | 无 | 当前有标签 head gradient | **是** |
| 额外开销 | 两路强视图 forward | 四路轻量评分 + 两路正常 forward | **是** |

简化流程如下：

```text
                         ┌─ candidate 1 ─ pseudo-gradient g1 ─ utility U1
                         ├─ candidate 2 ─ pseudo-gradient g2 ─ utility U2
unlabeled weak image ────┼─ candidate 3 ─ pseudo-gradient g3 ─ utility U3
                         └─ candidate 4 ─ pseudo-gradient g4 ─ utility U4
                                              │
labeled batch ─ supervised CE+Dice ─ head gradient gL
                                              │
                                     Top-2 utility ranking
                                              │
                         selected two views ─ normal UniMatch update
```

## 6. 可以概括为哪些创新点

### 创新点一：将随机强增强采样改写为在线任务效用选择

原 UniMatch 默认每一次随机强增强都具有同等训练价值。UtilityMatch 将强增强视图看成
一组待选择的训练动作，并用它们对真实分割任务产生的即时优化贡献进行排序。这比单纯
增加新的增强算子更接近一个定义明确的方法机制。

### 创新点二：用真实标注梯度作为增强选择的任务锚点

方法不再使用模型自身的 confidence、entropy、JS 或 temporal stability 形成闭环判断，
而是利用训练中本来就存在的少量有标签样本定义参考方向。它选择的是“对有标签任务
优化方向有用的强视图”，而不是“最容易、最稳定或最自信的强视图”。

### 创新点三：保留困难增强，只排除优化方向冲突

高不一致强视图可能是有用 hardness，也可能是破坏语义的噪声。UtilityMatch 不直接
惩罚不一致、不按风险降低 loss weight，也不修改 pseudo target；只判断其伪监督梯度
是否与真实任务方向兼容。因此它有机会保留困难但可学习的增强，同时减少方向冲突的
训练信号。

### 创新点四：最小干预、可插拔的实现

候选评分只作用于最终输出卷积，使用 detached feature，并隔离 BN running statistics；
选中后仍走完整原始 UniMatch 训练路径。模型结构、teacher、target、loss 和自训练更新数
均不变，使得最终差异可以较清楚地归因于 strong-view selection。

其中前三点是方法层面的主要贡献，第四点主要是实现与实验设计贡献，不能单独作为论文
核心创新。

## 7. 与前序方法和失败方向的区别

- **不同于 temporal-v2/UATS/ST++ 路线**：不建立时间概率库、不使用 MC-dropout，
  也不依据跨 checkpoint 稳定性筛选伪标签。
- **不同于 TPC-B**：不把高 weak/strong JS 当成风险并削弱该视图；高不一致视图只要
  梯度方向有用，仍可被选中。
- **不同于 TCR**：不使用 exact-strong prediction 替换 transported weak target，
  pseudo label 的类别和置信度均保持原样。
- **不同于普通 confidence curriculum**：置信度仍只负责像素 eligibility，不负责
  候选视图排名。
- **不同于 aggregate gradient balancing**：当前机制在参数更新前选择具体增强视图，
  并不是在采样完成后重新组合整体 supervised/unsupervised loss 的下降方向。

因此，本方法最准确的定位是：

> **label-gradient-anchored transformation selection for weak-to-strong
> semi-supervised segmentation**。

## 8. 它是否在“提高伪标签质量”

严格来说，当前 UtilityMatch **没有生成更准确的新伪标签，也没有修改伪标签内容**。
它改善的是伪标签被使用的条件：相同 pseudo label 只通过更有任务效用的强增强视图进入
训练。因此论文中宜使用：

- “提高伪监督信号的优化质量”；
- “减少与有标签任务冲突的增强—伪标签组合”；
- “选择任务一致的 strong-view supervision”。

在没有直接测量 pseudo-label correctness 之前，不宜写成“UtilityMatch 提高了伪标签
本身的准确率”。从最初的两条研究主线看，它主要属于第 1 条——**数据增强选择**，同时
间接改善第 2 条——**伪监督的有效质量**。

## 9. 当前可以与不可以做的论文主张

### 现在可以陈述

1. 提出一种由 clean labeled gradient 引导的 strong-view 在线选择机制；
2. 在不改变 U-Net 和伪标签路径的情况下，把 UniMatch 随机双强视图替换为
   gradient-utility Top-2；
3. 方法直接评估实际采样变换的优化方向，而不是依赖 confidence/stability proxy；
4. 候选评分使用 head-only gradient 和 detached feature，避免保存四套完整反向图。

### 完整结果出来前不能陈述

1. UtilityMatch 一定优于 UniMatch 或 temporal-v2；
2. gradient utility 与真实 pseudo-label correctness 存在稳定相关性；
3. 方法可以改善最差病例或避免后期退化；
4. 方法具有跨 seed、跨数据集、跨 SSL baseline 的泛化性；
5. 方法达到 SOTA 或已经具备 CVPR 级证据。

当前只完成一个 seed 1337，因此即使结果提高，也只能作为第一阶段支持证据。

## 10. 论文实验应如何证明这项创新

主实验至少应区分三类公平性：

1. **相同更新次数**：原 UniMatch 和 UtilityMatch 都从固定 Pre10000 权重开始，
   自训练 30,000 次；
2. **相同计算量控制**：UtilityMatch 多出候选评分计算，后续需要 equal-wall-clock 或
   equal-FLOPs UniMatch；
3. **选择机制控制**：生成并评分相同四路候选但最终随机选两路，用于排除候选池和额外
   forward 本身的影响。

建议主要报告：

- 10 个 test patient 的 Dice、HD95 和逐病例配对差值；
- 最差病例及严重退化病例数量；
- 所有候选与被选候选的 `U_k` 分布、正 utility 比例；
- utility 与有标签候选真实 Dice degradation 的关系；
- 至少 3 个 matched seeds 的均值、标准差和 patient-level paired interval；
- 后续在第二个 SSL baseline 或第二个医学分割数据集上的可迁移性。

如果结果为正，最关键的消融顺序应是：随机 2-view、4-candidate random Top-2、
Utility Top-2；随后再比较 signed projection、cosine alignment 和仅梯度符号等评分方式。

## 11. 可直接用于论文/汇报的方法描述

> UniMatch 通过双强视图一致性提高半监督分割性能，但其强增强视图由固定分布随机采样，
> 无法区分有用困难样本与推动模型偏离真实任务的增强噪声。为此，我们提出
> UtilityMatch：一种由有标签任务梯度引导的在线强视图选择方法。对于每个未标注
> mini-batch，我们从原 UniMatch 增强分布生成四个候选视图，并计算每个候选的置信度
> 掩码伪监督梯度在当前有标签监督梯度方向上的有符号投影。效用最高的两个候选随后通过
> 完整 Student 网络执行原 UniMatch 更新。该方法不修改分割架构、EMA 伪标签、置信度
> 筛选或损失组合，而是将随机增强采样转化为真实标注锚定的任务效用选择。

## 12. 建议的模型名称与标题表达

模型名可保留 **UtilityMatch**，全称建议为：

> **UtilityMatch: Clean-Gradient Utility Guided Strong-View Selection for
> Semi-Supervised Medical Image Segmentation**

若后续实验证明其跨 baseline 泛化，再把标题中的 UniMatch 扩展为通用 weak-to-strong
consistency；如果只在当前 PROMISE12 UniMatch 上有效，则不应过度宣称通用性。

## 13. 代码对应关系

- 原 UniMatch 训练流程：[`../../code/train_unimatch.py`](../../code/train_unimatch.py)
- UtilityMatch 完整训练：[`../../code/train_utilitymatch.py`](../../code/train_utilitymatch.py)
- 梯度投影与 Top-2 原语：[`../../code/utilitymatch.py`](../../code/utilitymatch.py)
- 锁定实验协议：[`protocol.md`](protocol.md)
- 服务器启动脚本：[`../../run_utilitymatch_5090.sh`](../../run_utilitymatch_5090.sh)
- 测试脚本：[`../../test_utilitymatch_5090.sh`](../../test_utilitymatch_5090.sh)

## 14. FrontierMatch：增强强度与伪标签覆盖率的统一扩展

GuardedUtilityMatch 从约 0.816 恢复到约 0.828，说明重新引入原始强增强能够追回
性能，但“最多选择一条原始强增强”的来源配额仍可能限制有效监督。FrontierMatch
不再固定保留某一候选来源，而是在原双强视图结构的每条独立视图流内，同时构造三种
增强—伪标签策略：

1. `stable`：p01--p99 亮度单位、置信度阈值 0.95；
2. `coverage`：校准与原始亮度单位的中点、置信度阈值 0.90；
3. `reliable`：原始绝对亮度单位、置信度阈值 0.98。

同一视图流内的三种策略共享随机对比度、亮度方向、模糊参数和 CutMix 映射，因此
utility 的差异主要反映增强强度与伪标签覆盖策略，而不是三次无关随机采样。对第
`k` 个联合策略仍使用：

```text
u_k = <g_k, g_L> / ||g_L||
```

每条视图流分别选择 utility 最大的策略，最终仍得到两条独立强视图。被选策略只有在
`u_k > 0` 时才进入强一致性损失；全负状态不重新分配权重，只保留原 feature
perturbation 分支。这样两条视图都可以选择新的强策略，但任何负迁移强监督仍被阻断。

FrontierMatch 不修改 PROMISE12 数据、35/5/10 划分、前七病例 191 slices、原
PreTrain、U-Net、EMA、weak pseudo-label 生成器、feature 分支 0.95 阈值、优化器、
验证、checkpoint 或测试逻辑。其核心研究问题是：相比固定增强和固定置信度阈值，
有标签任务梯度能否在线选择更有效的增强难度—伪标签可靠性平衡点？

代码对应关系：

- 训练入口：[`../../code/train_frontiermatch.py`](../../code/train_frontiermatch.py)
- 冒烟测试：[`../../code/test_frontiermatch_smoke.py`](../../code/test_frontiermatch_smoke.py)
- 锁定协议：[`../frontiermatch/protocol.md`](../frontiermatch/protocol.md)
- 服务器启动：[`../../run_frontiermatch_5090.sh`](../../run_frontiermatch_5090.sh)
- 测试入口：[`../../test_frontiermatch_5090.sh`](../../test_frontiermatch_5090.sh)
