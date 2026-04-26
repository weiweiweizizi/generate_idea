# LQ Laterality Contrast Token Design

**状态**: 已批准实现

**日期**: 2026-04-21

---

## 1. 背景

当前 `v28 side-aware pooling` 已验证：

- 单靠后段 pooling 无法修复 shared trunk 中已经混入的 side / dataset 信息
- `side_from_side_rep_acc` 没有优于 `v26`
- `dataset_from_side_rep_acc` 明显上升
- `z_free` 与 `z_side` 的线性可解释度显著变高

因此下一步不再继续做 “只改 pooling” 的弱结构修补，
而是直接给 side branch 注入显式 laterality 结构先验。

---

## 2. 目标

在不改 shared trunk、free branch、private branch、FSQ 配置和训练损失接口的前提下，
把 early side path 的读出改成显式左右差分 token：

- `around-left - around-right`
- `mouth-left - mouth-right`

目标不是立刻优化所有指标，而是回答一个更关键的问题：

> 当 side branch 被强制读取“左右不对称”而不是“绝对局部块”时，
> side 语义是否会更集中在 side rep 中，同时减少 dataset leakage 和 free/side 纠缠。

---

## 3. 设计约束

- 保持 `v28` 的 shared trunk 不变
- 保持 `residual_fsq` quantizer、不改 level 结构
- 保持 `side_basis_count = 4`
- 保持 `side_z_dim = 32`
- 保持 canonical analysis 口径不变
- 新结构必须通过 checkpoint config 可重建，不能依赖隐式源码状态

---

## 4. 主方案

### 4.1 新 side pooling 模式

新增一个显式 side pooling 模式，例如：

- `side_pooling = fixed_region2_contrast`

其输入仍为：

- `side_feats = side_adapter(feats)`，shape `[B*T, 32, 15, 15]`

### 4.2 固定区域定义

沿用上一轮 `15 x 15` 上的四块近似映射：

- `around_left  = [0:3, 0:3]`
- `around_right = [3:6, 3:6]`
- `mouth_left   = [6:10, 6:10]`
- `mouth_right  = [10:15, 10:15]`

### 4.3 contrast token 构造

先对每个 block 做平均池化，得到四个 `[B*T, 32]` token：

- `around_left_token`
- `around_right_token`
- `mouth_left_token`
- `mouth_right_token`

再显式构造两个差分 token：

- `around_contrast = around_left_token - around_right_token`
- `mouth_contrast = mouth_left_token - mouth_right_token`

最终 side readout 为：

- `concat([around_contrast, mouth_contrast], dim=1)`，shape `[B*T, 64]`

再接：

- `side_head(64 -> hidden -> side_z_dim=32)`

---

## 5. 预期

若该结构有效，应优先看到以下信号：

- `side_from_side_rep_acc` 上升
- `side_from_free_rep_acc` 不同步上升
- `dataset_from_side_rep_acc` 下降，或至少不再继续升高
- `ortho_linear_r2_free_to_side` / `ortho_linear_r2_side_to_free` 下降

允许的代价：

- `val_recon` 小幅波动

不接受的结果：

- side probe 没提升，但 `dataset_from_side_rep_acc` 继续走高
- free/side 耦合继续升高

---

## 6. 实施范围

本轮仅包含：

- `scripts/lq/model/distnet.py`
- 新的训练 preset
- 必要的 smoke test
- 必要时更新研究日志

如果这轮仍失败，下一步再考虑更激进的 trunk-level 明确分流，而不是继续堆 readout trick。
