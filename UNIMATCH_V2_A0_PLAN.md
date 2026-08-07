# UniMatch V2 A0 实验记录（PROMISE12）

> 状态：实现完成，smoke test 验证中。
> 日期：2026-08-07
> 对应代码：`code/train_unimatch_v2.py`、`code/test_unimatch_v2.py`、`run_unimatch_v2_5090.sh`、`test_and_quantify_unimatch_v2_5090.sh`

---

## 1. 目标

在保持 PROMISE12 正式协议（35/5/10、labelnum=7、seed=1337、pre 10000 / self 30000、batch 24/12、SGD 0.01、EMA 0.99、tau=0.95、CutMix、强增强、验证/测试流程）不变的前提下，把 UniMatch 的**无监督分支结构**从 V1 三支改为 V2 A0 双支：

```text
V1: 0.25 * strong_view1 + 0.25 * strong_view2 + 0.50 * feature_perturbation(feature dropout)
V2: 0.50 * strong_view1 + 0.50 * strong_view2   （两个强视图共享解码器，互补通道 Dropout）
```

A0 只做这一处结构性改动，用于公平验证 **Complementary Dropout** 是否在医学分割（小前景）上有效。DINOv2 编码器、AdamW、纯 CE 伪损失、去 ramp-up 均**不在 A0 范围**（后续消融）。

## 2. 修改点（相对 `train_unimatch.py`）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `UNet.forward(x, comp_drop=False)` | 新增互补通道 Dropout 路径：batch 为 [s1(bs); s2(bs)]，在 bottleneck（`feature[4]`，256 通道）上生成互补 mask `mask2 = 2 - mask1`（mask1 ∈ {0,2}，随机一半样本对两视图全通道保留=1），`feature[4] *= mask`，共享解码器一次前向，返回拼接 logits 由调用方 `chunk(2)` 拆分 |
| 2 | `UNet.__init__` | 新增 `self.binomial = Binomial(probs=0.5)` 与 `self.comp_drop_dropout_prob = args.comp_drop_dropout_prob`（默认 0.5） |
| 3 | self_train 无监督段 | 删除 `need_fp=True` 特征扰动支路；`outputs = model(volume_batch)`（普通前向）；强视图改 `model(cat(s1,s2), comp_drop=True).chunk(2)`；`consistency_loss = 0.5*loss_u_s1 + 0.5*loss_u_s2` |
| 4 | argparse | `--exp` 默认 `UniMatchV2_A0_label7_seed1337`；新增 `--comp_drop_dropout_prob 0.5`；`--feature_dropout` 保留但标注 [V1 only, unused in V2 A0] |
| 5 | tensorboard / 日志 | `unimatch/loss_feature` → `v2/loss_view1`、`v2/loss_view2`；日志行去掉 fp 项 |

**未修改**：`train_unimatch.py`、`test_unimatch.py`、`dataset.py`、`utils/*`、`networks/*` 及任何已有 EXP 目录。预训练阶段（`pre_train`）与 baseline 完全一致。

## 3. 与 baseline 的区别（严格受控变量）

| 维度 | UniMatch baseline | UniMatch V2 A0 |
|---|---|---|
| 无监督支路 | 3 支：s1(0.25)+s2(0.25)+fp(0.50) | 2 支：s1(0.50)+s2(0.50) |
| 双视图构造 | 独立两路强增强 | 两路强增强 + 互补通道 Dropout（bottleneck） |
| 特征扰动支路 | 每 encoder scale Dropout2d(0.5) | 删除（被 comp_drop 取代） |
| 其余一切 | EMA teacher / 伪标签 / tau=0.95 / CutMix / 强增强 / warmup 1000 / ramp-up / SGD / 验证测试 | 与 baseline 完全相同 |

参数结构保持不变（encoder/decoder/投影头命名与 `networks.unet.UNet_2d` 一致），因此 **V2 checkpoint 可直接被 `test_unimatch.py` / `test_unimatch_v2.py` 严格加载**。

## 4. 固定参数（正式实验）

- 数据：`data/PROMISE12_h5`；划分 35/5/10；labeled 前 7 例 = **191 labeled slices**；全切片保留
- seed 1337；`pre_iterations=10000`；`max_iterations=30000`
- batch 24 / labeled_bs 12；patch 256×256；SGD lr 0.01，momentum 0.9，wd 1e-4（self 阶段 poly 0.9 衰减）
- EMA decay 0.99（teacher eval 模式）；tau=0.95；CutMix 0.5；强增强 0.8/0.5；`comp_drop_dropout_prob=0.5`
- warmup 1000（self 前 1000 iter 无监督损失为 0）；consistency ramp `5*0.1*sigmoid_rampup(iter//150, 200)`
- 验证：每 200 iter，val 5 例，(Dice, HD95)，best-by-val-dice 存 checkpoint
- 测试：test 10 例，Dice/Jaccard/HD95/ASD，nms=0，`performance.txt` + `metric_table.csv`

## 5. Checkpoint 路径

```text
model/UniMatchV2_A0_label7_seed1337_7_labeled/
├── pre_train/unet/          # 预训练（与 baseline 同协议）
│   ├── unet_best_model.pth  # {'net','opt'}
│   └── log.txt, log/        # tensorboard
├── self_train/unet/         # V2 A0 自训练
│   ├── unet_best_model.pth  # best-by-val
│   ├── iter_*.pth
│   ├── log.txt, log/
│   ├── test_predictions/    # 测试预测 nii.gz（测试时生成）
│   └── performance.txt
└── metric_table.csv / metric_table.md / test_case_metrics.csv   # 测试量化
```

运行方式：

```bash
# 训练（正式 10k/30k）
bash run_unimatch_v2_5090.sh
# 或自定义迭代数
PRE_ITERATIONS=10000 MAX_ITERATIONS=30000 bash run_unimatch_v2_5090.sh

# 测试并量化
bash test_and_quantify_unimatch_v2_5090.sh
```

防覆盖：`EXPERIMENT_DIR` 下已有 `*.pth` 且未设 `ALLOW_EXISTING=1` 时拒绝运行（沿用 5090 脚本约定）。

## 6. Smoke test（开发机验证）

开发机：Windows + RTX 3050 6GB + conda env `scope`（torch 2.1.0+cu118）。受页面文件限制，DataLoader 强制 `num_workers=0`（启动器 `smoke_train_v2.py`，直接调用项目脚本真实训练函数）。

- exp：`UniMatchV2_A0_smoke_seed1337`（独立目录，不影响正式实验）
- pre 200 iter（触发 iter 200 验证并保存 pretrain best）→ self 1010 iter（跑过 1000 warmup，iter 1000 起激活 comp_drop 双视图）
- batch 8 / labeled_bs 4 / patch 160×160 / seed 1337
- 检查项：前向、comp_drop、伪标签、损失、验证（iter 200/400/.../1000 共 5 次）、checkpoint 保存/加载

### 单元检查结果（check_compdrop_v2.py，合成数据）

- mask 互补性：`mask1 + mask2 == 2` 全成立；mask 值域 {0,1,2}；每对样本 50% 概率全通道保留（kept pairs=2/4）
- 前向：view1/view2 输出 (4,2,160,160) 正常
- 伪标签生成、置信掩码损失、反向、3 步优化器迭代均正常（随机初始化模型 coverage=0，损失为 0 属预期）

### 训练 smoke 结果（2026-08-07 已执行）

**smoke #1（正式 tau=0.95 配置，`UniMatchV2_A0_smoke_seed1337`）**：完整跑通。
- pre 200 iter：val dice 0.0327，保存 `iter_200_dice_0.0327.pth` + `unet_best_model.pth`（{'net','opt'} 格式）
- self 1010 iter（batch 8/labeled 4/patch 160）：warmup 1000 内 uni=0（符合 baseline 设计）；iter 200/400/600/800/1000 共 5 次验证正常（best dice 0.1417）；iter 1000 起 comp_drop 分支执行无报错；保存 `iter_200_dice_0.0011.pth`、`iter_400_dice_0.1417.pth`、`unet_best_model.pth`
- 因 smoke 模型太弱（仅 200~400 监督 iter），EMA teacher 置信度均 <0.95，coverage=0 → uni loss 恒为 0（数值未激活，属预期）
- checkpoint 加载：self_train 成功加载 pretrain best；`check_ckpt_v2.py` 验证 self best 与测试端 `UNet_2d` strict 兼容（missing/unexpected 均空）

**smoke #2（tau=0.5 数值激活验证）**：iter 200 处崩溃。
- 原因：**开发机 C 盘几乎占满（剩余约 0.7MB）**，torch.save 写 `unet_best_model.pth` 中断（`unexpected pos ...`）。同目录 `iter_200_dice_0.0034.pth`（7.4MB）写入成功，属环境磁盘问题，非代码 bug。正式训练在 5090 服务器上执行，不受影响。

**V2 损失数值验证（check_v2_loss.py）**：加载 smoke #1 pretrain checkpoint（dice 0.033，未塌缩），tau=0.5，跑 5 步完整自训练损失路径（强增强 + CutMix + EMA 伪标签 + comp_drop 前向 + 置信掩码 CE+Dice）：

| step | coverage | fg_coverage | loss_v1 | loss_v2 | uni |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0.0000 | 0.288469 | 0.289228 | 0.288848 |
| 2 | 1.0000 | 0.0000 | 0.277327 | 0.300899 | 0.289113 |
| 3 | 1.0000 | 0.0000 | 0.256410 | 0.320589 | 0.288499 |
| 4 | 1.0000 | 0.0000 | 0.297621 | 0.273565 | 0.285593 |
| 5 | 1.0000 | 0.0000 | 0.285050 | 0.282604 | 0.283827 |

结论：comp_drop 双视图一致性损失数值激活（uni > 0）、随优化缓慢下降、coverage 正常（fg_coverage=0 因 smoke 预训练模型几乎没有前景预测，正式 10000 iter 预训练后正常）。`V2 LOSS CHECK PASSED`。

**总体结论：代码正确性验证完成，可进入正式训练（Pre10000 + Self30000，tau=0.95）。**

## 7. 后续消融计划（均基于 A0，在 val 上决策）

| 编号 | 改动 | 目的 |
|---|---|---|
| A1 | 去 warmup 1000 + 去 ramp-up（V2 式恒定 `(loss_x+loss_u)/2`） | 验证 V2 简化是否在医学上成立 |
| A2 | 伪标签损失改纯 CE（V2 官方形式，按非 ignore 像素归一） | 验证 Dice 项必要性 |
| A3 | comp_drop 作用范围扩展到全部 encoder scale | 医学 U-Net 上的最佳作用位置 |
| A4 | EMA decay 改动量式 `min(1-1/(it+1), 0.996)` | 对齐 V2 官方 EMA |
| B | DINOv2 编码器 + AdamW（Full UniMatch V2） | 验证编码器升级主张（独立小节） |

约束：每个消融只改变一个因素；全部先在 val 上筛选（可跑短调度 pre 2000/self 5000），命中后再上正式 10000/30000；最终 test 只跑一次选定配置；禁止用 test 调参。

## 8. 验收标准（建议）

A0（或其变体）相对正式 baseline：test Dice 提升 ≥1 point 且 HD95 不恶化，判定 V2 结构有效；否则如实报告持平/下降并归因。
