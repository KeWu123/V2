# CalibratedUtilityMatch 完整实验

## 推荐结论

优先运行本实验，不先运行 POS。原因不是猜测超参数，而是已经完成内容级数据核验：
桌面 1427 个 H5 与 `KeWu123/data` 的 Git LFS SHA256 全部一致，而这份数据是
非负缩放域；原 UtilityMatch 的强增强却按 z-score 假设使用绝对亮度偏移
`[-0.25, 0.25]`。

本实验不改数据，只修复 UtilityMatch 的候选强增强，并保留负效用分支弃权。

## 相对 UtilityMatch 的变化

1. 亮度校准：

   ```text
   old: brightness = Uniform(-0.25, 0.25)
   new: brightness = Uniform(-0.25, 0.25) * sampled_p01_p99_range(slice)
   ```

2. 有符号安全门：Top-2 中仅 `utility > 0` 的强分支进入损失；拒绝的 0.25
   权重不重新分配，Feature perturbation 的 0.5 权重不变。

其余完全保持：原始 PreTrain、35/5/10、first7=191、seed1337、warm-up1000、
Self30000、U-Net、EMA train mode、0.95 confidence、NMS、contrast 0.5–1.5、
blur 0.1–2.0、CutMix、K=4/Top-2、SGD/poly LR、Student validation/test。

## 服务器运行

前台显示进度：

```bash
cd /home/aiteam/zhengtaoma/Baseline
bash run_utilitymatch_calibrated_5090.sh
```

若原始权重文件夹不在自动搜索位置：

```bash
ORIGINAL_UNIMATCH_DIR=/绝对路径/UniMatch_35_5_10_Pre10000_Self30000_label7_seed1337_7_labeled \
  bash run_utilitymatch_calibrated_5090.sh
```

训练中或训练完成后测试 validation-best Student：

```bash
bash test_utilitymatch_calibrated_5090.sh
```

后台运行：

```bash
DETACH=1 bash run_utilitymatch_calibrated_5090.sh
```

## 强制数据身份

运行前不仅检查数量，还会计算：

- 五个列表 bundle SHA256：
  `e0bd27c2d40977ab97b3059fecff965f1f270b7139ad01ad679b85e86ccf41e3`；
- 实际训练/验证/测试会读取的 955 个 H5 bundle SHA256：
  `332e491c9022a4542be148c770c56f00f5817a2141cce5d5795ebc91cbb6fe73`。

它们对应 `KeWu123/data@e58bb4db80006862a92e977b8525f513478c631a`。任何
数据、list、H5 内容或路径读取差异都会在训练前中止；脚本不会重写或转换数据。

## 结果判定

- 首先检查终端是否打印正确仓库 commit、两个 bundle hash、first7=191、原始
  PreTrain SHA256；
- `config.json` 必须为 `H-CALIBRATED-UTILITYMATCH`；
- 第 1--1000 步是原样监督 warm-up，因此与 UtilityMatch 一致是预期行为；
- 第 1001 步必须出现 `CALIBRATED-AUG active`，之后终端必须持续出现
  `CALIBRATED-AUG` 与 `UTILITY-GATE active`；缺任一标志都视为方法未执行；
- `calibrated_augmentation_trace.csv` 必须记录实际 p01--p99 范围和亮度位移；
- `utility_gate_trace.csv` 应显示非正所选分支被拒绝；
- 主指标为 validation-best Student 的固定十病例 test Dice；
- 要求超过有效 UtilityMatch 0.838397，并优先超过 temporal-v2 0.841027；
- 必须检查 Case34、Case05、Case09 及十病例配对差值，不能由单病例独占提升；
- 只跑 seed1337，因此结果只能判断下一步方向，不能声明统计显著。

若本实验失败，下一步再实现 POS 作为已发表强对照；不要通过修改数据、缩小测试集
或改变 PreTrain 来补救。
