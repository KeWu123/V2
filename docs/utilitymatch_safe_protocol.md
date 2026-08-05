# SafeUtilityMatch：下一步单变量实验

## 结论先行

当前最值得运行的下一步不是继续增加新的不确定性、时间库或网络模块，而是修复
UtilityMatch 与其自身定义不一致的行为：候选效用是有符号的，负值表示该强增强
产生的伪监督梯度与干净标注梯度冲突，但原实现即使四个候选全部为负仍强制训练
Top-2。

SafeUtilityMatch 保持当前 UniMatch/UtilityMatch 的完整基础架构，只加入严格的
分支弃权：所选强视图仅当效用严格大于 0 时进入损失。

## 科学假设

> UniMatch 退化并非因为强增强普遍无效，而是因为固定双强分支在某些批次强制
> 吸收与标注任务冲突的伪梯度。让增强分支根据干净标注梯度进行有符号弃权，可在
> 不改变模型和伪标签生成器的条件下降低错误确认偏差。

这同时连接了最初的两个研究方向：候选空间属于数据增强，是否允许该候选进入
训练则是伪监督质量控制。

## 方法

原 UtilityMatch 对四个真实采样的 UniMatch 强视图计算

\[
u_k = \frac{g_k^\top g_l}{\lVert g_l\rVert_2 + \epsilon},
\]

其中 \(g_l\) 是当前有标注批次在输出头上的梯度，\(g_k\) 是候选强视图伪损失
在同一参数块上的梯度。仍按 \(u_k\) 选取 Top-2，记为 \(s_1,s_2\)，但增加
无参数门控

\[
m_i=\mathbb{1}[u_{s_i}>0].
\]

最终一致性损失为

\[
L_u=0.25m_1L_{s_1}+0.25m_2L_{s_2}+0.5L_{fp}.
\]

缺失分支的 0.25 权重不重新分配。这样当证据不足时，伪监督总强度自动下降；若
重新归一化，会把仅剩的一个正分支权重从 0.25 放大到 0.5，同时混入新的损失
加权变量。Feature perturbation 分支完全不变。

实现仍前向计算两个所选强视图，以维持原 UtilityMatch 的候选结构和计算路径；
被拒绝分支通过低于 0.95 的置信掩码得到严格为零的 CE+Dice 伪损失。

## 唯一变化与固定项

固定不变：

- U-Net、参数量、初始化方式；
- `PROMISE12_h5_training_source`，35/5/10，`train_slices.list` 前七病例共
  191 slice；
- 原始 `UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled`
  文件夹中的 PreTrain 权重；
- seed 1337，batch 24/12，warm-up 1000，SelfTrain 30000；
- EMA train-mode 伪标签、阈值 0.95、NMS、增强算子及其概率；
- 四候选、Top-2、输出头梯度效用、Feature Dropout；
- SGD、poly LR、验证 Student、checkpoint 与测试逻辑。

唯一干预是 `selected_utility > 0`。原始 `train_utilitymatch.py`、
`utilitymatch.py` 和原运行脚本没有被覆盖。

## 服务器命令

前台运行并直接显示 tqdm、loss 和验证结果：

```bash
cd /home/aiteam/zhengtaoma/Baseline
bash run_utilitymatch_safe_5090.sh
```

若原始 UniMatch 文件夹不在脚本自动搜索的三个固定位置：

```bash
cd /home/aiteam/zhengtaoma/Baseline
ORIGINAL_UNIMATCH_DIR=/绝对路径/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled \
  bash run_utilitymatch_safe_5090.sh
```

后台运行：

```bash
DETACH=1 bash run_utilitymatch_safe_5090.sh
```

训练中或训练后测试当前 validation-best Student：

```bash
bash test_utilitymatch_safe_5090.sh
```

测试某个显式阶段权重：

```bash
UTILITYMATCH_DIR=/home/aiteam/zhengtaoma/Baseline/model/SafeUtilityMatch_<时间>_7_labeled \
CHECKPOINT=/home/aiteam/zhengtaoma/Baseline/model/SafeUtilityMatch_<时间>_7_labeled/self_train/unet/iter_<步数>_dice_<数值>.pth \
  bash test_utilitymatch_safe_5090.sh
```

## 运行时证据

启动器在训练前会中止任何不一致：错误目录名、非 35/5/10 列表、非 940 训练
slice、前七病例不等于 191、H5 缺失、非原始 PreTrain 格式或错误 GPU。终端会
打印数据绝对路径和两个固定权重的 SHA256。

除原训练日志、TensorBoard 和 `config.json` 外，新版本写出
`utility_gate_trace.csv`，每 20 步记录四个效用、Top-2 索引、两个门值和有效
强分支数；`training_summary.json` 汇总有效门比例与全拒绝批次数。

## 判定标准

本次只跑 seed 1337，因而它是方向筛选，不是统计显著性证明。

- 机制正确：负值或零值所选分支的对应 `s1/s2` 损失必须为 0，正分支不变；
- 性能成功：validation-best Student 的十病例 test Dice 超过有效 UtilityMatch
  0.838397，并优先要求超过 temporal-v2 0.841027；
- 病例安全：不能只依赖单病例（此前 Case34）贡献全部均值增益，应检查十病例
  配对差值和最差退化病例；
- 若门几乎从不触发，则该实验退化为原 UtilityMatch，不能宣称方法有效；
- 若门频繁触发但 test 仍下降，则“输出头瞬时梯度符号足以判定增强安全”被否定，
  不应继续调门限或叠加更多门，而应回到纯 UniMatch 的伪标签校准方向。

## 创新边界

梯度冲突、梯度匹配和多目标优化已有充分先例，因此不能宣称一般性的
“gradient conflict resolution”。可检验的具体贡献是：对 UniMatch 当次真实采样
的增强实例进行任务对齐评估，并将有符号效用解释落实为不重新归一化的增强弃权
机制。它比全局重加权更贴近本文“数据增强如何产生高质量伪监督”的问题，但在多
seed、多基线验证前仍只应称为候选方法，而非最终创新结论。
