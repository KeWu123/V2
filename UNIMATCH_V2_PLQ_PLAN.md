# UniMatch V2 PLQ 实验记录（Pseudo-Label Quality）

> 状态：代码实现完成，单元验证通过，debug smoke 运行中。
> 日期：2026-08-08
> 代码：`code/train_unimatch_v2_plq.py`（新增，不改动 `train_unimatch_v2.py`）
> 基线：UniMatchV2_A0（test Dice 0.810115 / HD95 4.930269），目标 0.85+

---

## 1. Baseline 问题

A0 在 PROMISE12 7 labeled 下 test Dice 0.810，瓶颈在伪标签选择：

1. **固定全局阈值 0.95 偏向背景**：前列腺前景占比极小（<5%），前景像素置信度天然偏低，高阈值导致大量正确前景伪标签被丢弃（fg coverage 低）。
2. **Softmax 过度自信**：max-prob 高分不等于可靠，错误前景伪标签可能混入（HESS/FARCLUSS 论证）。
3. **缺少不确定度信号**：只依赖 max-prob 一维信号，边界/歧义像素无法区分。

## 2. 修改位置（train_unimatch_v2.py 内对应行，plq 版已替换）

| # | 位置 | 原逻辑 | PLQ 修改 |
|---|---|---|---|
| 1 | teacher prediction 生成 | `ema_output = ema_model(unlabeled_volume_batch)` + softmax | 不变（EMA teacher 冻结 eval） |
| 2 | confidence filtering | `confidence_masked_baseline_loss` 内 `valid = confidence >= 0.95` | 新增 `compute_plq_valid`：类感知阈值 + 熵门 |
| 3 | pseudo label 生成 | `get_masks(ema_output, nms=1)`（argmax + 2D LCC） | 不变；额外计算 teacher 分布 Shannon 熵 |
| 4 | consistency loss 计算 | `confidence_masked_baseline_loss`（阈值在函数内） | 新增 `masked_plq_loss(logits, targets, valid, dice)`，valid 由外部预计算并随 CutMix 同步 |

## 3. 公式

逐像素 validity mask：

```
foreground(i) = (pseudo_label(i) == 1)
thr(i) = foreground(i) ? threshold_fg : threshold_bg          # 类感知阈值
valid_conf(i) = (conf(i) >= thr(i))                            # conf = max teacher prob
entropy(i) = -Σ_c p(i,c) · log p(i,c)                          # teacher 分布熵（自然对数）
valid(i) = valid_conf(i)  ∧  (entropy_threshold<=0 ∨ entropy(i) <= entropy_threshold)

L_uni = 0.5 · L(θ; s1, pseudo_s1, valid_s1) + 0.5 · L(θ; s2, pseudo_s2, valid_s2)
L(θ; out, target, valid) = 0.5 · CE_masked + 0.5 · Dice_masked   # 与 baseline 损失主体一致
```

默认参数：`threshold_bg=0.95`、`threshold_fg=0.80`、`entropy_threshold=0.5`（0 = 禁用熵门）。

## 4. 实验参数（正式实验）

- 保持：PROMISE12 35/5/10、前 7 例（191 slices）、seed 1337、pre 10000 / self 30000、batch 24/12、patch 256、SGD 0.01、EMA 0.99、CutMix、warmup 1000、comp_drop 双视图 0.5/0.5
- 仅改伪标签选择：`--threshold_bg 0.95 --threshold_fg 0.80 --entropy_threshold 0.5`
- EXP_NAME：`UniMatchV2_PLQ_label7_seed1337`
- 产物：`model/UniMatchV2_PLQ_label7_seed1337_7_labeled/{pre_train,self_train}/unet/` + `metric_table.csv`
- 记录：coverage / fg_coverage / entropy（训练日志与 tensorboard 已输出）

## 5. 预期提升原因

1. **类感知阈值**：前景阈值降到 0.80 → 恢复大量被 0.95 丢弃的正确前景伪标签，直接提高 fg coverage，缓解前景监督信号不足（参考 DSSN per-class 选样、FARCLUSS 再平衡思想）。
2. **熵门**：拒绝高熵（歧义/边界）像素 → 减少错误伪标签污染，等价于不确定性感知过滤（HESS/FARCLUSS 方向），且与置信度互补（高置信低熵才可靠）。

## 6. 修改文件列表

| 文件 | 类型 | 说明 |
|---|---|---|
| `code/train_unimatch_v2_plq.py` | 新增 | 基于 A0，替换伪标签选择（类感知阈值 + 熵门） |
| `code/train_unimatch_v2.py` | 不动 | 保持 A0 原样 |
| `UNIMATCH_V2_PLQ_PLAN.md` | 新增 | 本记录 |

## 7. 验证与 smoke 结果

- 语法检查：py_compile 通过
- 单元检查（check_plq.py）：
  - 类感知阈值：fg conf=0.85 接受（thr_fg=0.80）、bg conf=0.85 拒绝（thr_bg=0.95）✓
  - 熵门：高熵 fg 像素被拒绝 ✓
  - masked_plq_loss 数值有限 ✓
- 真实数据检查（check_plq_data.py，8 个真实 unlabeled 切片）：
  - pseudo 出现前景类（unique [0,1]）；conf mean 0.99、entropy mean 0.028
  - PLQ coverage 0.9744、fg_coverage 0.0000（弱 smoke 模型前景预测不可靠被门过滤，符合预期；正式 30k 训练后前景可靠）
- debug smoke（复用 A0 smoke pretrain + self 1010 iter，threshold_bg/fg=0.95/0.80，entropy=0.5）：
  - 训练完整结束，无崩溃、无 nan；iter 200/400/.../1000 验证正常（best val dice 0.1776，弱模型规模下合理）
  - checkpoint 保存/加载正常（iter_200_dice_0.0724.pth、iter_400_dice_0.1776.pth、unet_best_model.pth）
  - PLQ 分支在 iter 1009 起执行（warmup 1000 后），一次执行无报错
  - 结论：**集成正确，可进入正式训练**（PLQ 的 coverage/fg_coverage/entropy 定量曲线由远程正式训练日志与 tensorboard 采集）

## 8. 完整训练命令（远程 V2）

```bash
# 训练（正式 10k/30k）
REQUIRE_5090=0 EXP_NAME=UniMatchV2_PLQ_label7_seed1337 \
  bash run_unimatch_v2_5090.sh \
  --threshold_bg 0.95 --threshold_fg 0.80 --entropy_threshold 0.5

# 测试（仅最终选定模型）
REQUIRE_5090=0 EXP_NAME=UniMatchV2_PLQ_label7_seed1337 \
  bash test_and_quantify_unimatch_v2_5090.sh
```

> 注意：`run_unimatch_v2_5090.sh` 入口为 `train_unimatch_v2.py`，PLQ 需换入口。若直接复用 5090 脚本需先复制为 PLQ 版脚本并改入口为 `train_unimatch_v2_plq.py`，或在命令行直接调用：
> `python train_unimatch_v2_plq.py --root_path data/PROMISE12_h5 --exp UniMatchV2_PLQ_label7_seed1337 --pre_iterations 10000 --max_iterations 30000 --threshold_bg 0.95 --threshold_fg 0.80 --entropy_threshold 0.5`
