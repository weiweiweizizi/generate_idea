# LQ Side Basis Representation Tightening Design

**状态**: 草稿

**日期**: 2026-04-20

---

## 1. 背景

`v26 early-branch` 已经验证了“更早分流”这个结构假设是成立的：

- `val_loss = 0.7623`
- `val_recon = 0.2873`
- `val_side_group = 0.5365`
- `side_from_side_rep_acc = 0.6500`
- `side_from_free_rep_acc = 0.3273`

相比 `v23`，`v26` 在重建、side/free probe 分离和 latent 解耦上都明显更好。

但当前仍有一个关键未解问题：

- `dataset_from_side_rep_acc = 0.8182`

这说明虽然 `side` 分支已经不再和 `free` 强耦合，但它依然在携带 dataset 信息。

当前 `v26` 的 side 分支仍然过于自由：

- `feats -> side_adapter -> side_pool -> side_head -> side_z`
- `side_z` 同时驱动：
  - side basis reconstruction
  - side supervision
  - analysis probe

问题在于：

- `side_z` 是一个 32 维 dense latent
- 模型可以一边完成 side 任务，一边把 dataset 信息藏在未显式约束的维度里

因此下一步不应优先继续加 adversarial，而应先收紧 side branch 的可用表达空间。

---

## 2. 目标

本设计的目标是让 side branch 的“可用表示”尽量落在显式 basis 激活上，而不是落在自由的 dense `side_z` 上。

第一版只做两个改动：

1. 把 `side` 的 canonical representation 从 pooled `side_z` 改为 pooled side basis expression
2. 把新 baseline 的 `side_z_dim` 从 `32` 收紧到小维度，推荐 `8`

成功标准：

1. `side_from_side_rep > side_from_free_rep` 继续成立
2. `dataset_from_side_rep` 相对可比基线下降
3. `val_recon` 不明显劣化，容忍上升不超过 `0.01`
4. `val_side_group` 不明显劣化，容忍上升不超过 `0.10`

这里的“可比基线”必须明确：

- 不是直接拿旧 `v26` summary 里基于旧 canonical rep 的数值对比
- 而是把 `v26` 的 best checkpoint 用**新的 canonical side rep 定义**重新跑一次 analysis

也就是说，这一轮的比较口径固定为：

- baseline: `v26` best checkpoint + new side-rep analysis rule
- candidate: `v27` best checkpoint + same analysis rule

---

## 3. 核心思路

### 3.1 Canonical side rep 改为显式 basis 表达

当前 `v26` 下，analysis 和部分训练语义实际仍然可以绕回 dense latent。

本设计中，side 分支的 canonical representation 改为：

- `side_path_usage`
- `side_path_coefficients`
- `side_path_representation = side_path_usage * side_path_coefficients`

group-level side representation 默认应使用：

- `masked_mean(side_path_representation, valid_mask)`

而不是：

- `masked_mean(side_z, valid_mask)`

这样可以把 side 分支的可解释性对齐到：

- 哪个 side basis 被使用
- 使用强度有多大

而不是一个难以解释的 hidden vector。

### 3.2 收紧 side latent 容量

在 early-branch baseline 上，把：

- `side_z_dim = 32`

收紧为：

- `side_z_dim = 8`

第一版不建议再小于 `8`，避免 side path 因容量过低直接崩掉。

这个改动的目的不是追求极限压缩，而是减少 `side_z` 继续作为 dataset shortcut 容器的空间。

---

## 4. 训练语义修改

### 4.1 Side group supervision

这里需要明确区分“已经存在的行为”和“本轮新增 delta”。

当前仓库里，`side_group` 训练监督已经是按 grouped side representation 做的，不需要在这一轮重构成另一套训练图。

本轮**不**以“重写 group-side supervision 路径”为主要目标。

当前 `side_group` 在训练上仍然保持：

- `group_side_rep = masked_mean_per_sequence(side_path_representation, valid_mask)`
- `group_side_logits = group_side_classifier(group_side_rep)`

因此，这一轮真正的新 delta 不是“把训练期 side_group 从 latent 改成 basis”，因为这件事在当前代码里已经基本成立。

这一轮新增目标是：

- 明确把 analysis / probe 的 canonical side rep 也对齐到同一套 basis expression
- 同时把 `side_z_dim` 收紧到 `8`

### 4.2 Side continuous supervision

frame-level `side_classifier(side_z)` 可以先保留，作为辅助项。

但第一版需要明确：

- 它不再是 side branch 的 canonical analysis representation
- 真正用于 probe 和 group-level side 判别的是 pooled side basis rep

这样可以减少“训练在 basis 上、分析在 latent 上”这种目标不一致。

---

## 5. Analysis 修改

analysis 里需要区分两层 side 表示：

1. `side_latent`
   - 仅作为内部诊断对象保留
2. `group_pooled_side_rep`
   - 作为 canonical side representation
   - 明确来自 pooled `side_path_representation`

probe 规则：

- `side_from_side_rep` 用 canonical `group_pooled_side_rep`
- `dataset_from_side_rep` 用 canonical `group_pooled_side_rep`
- side/free latent 线性 recoverability 仍然可以保留，但只作为诊断项

第一版不改 `free` 的 canonical rep，仍保持当前 `group_pooled_free_rep` 逻辑。

### 5.1 比较协议

为保证数值可比，评估协议固定如下：

- 数据集：`data/win20-step20/IMR,data/win20-step20/TT`
- split：沿用当前训练脚本的同一 subject split 规则
- checkpoint 选择：仍使用各自 run 的 `best.pt`
- 对比对象：
  - `v26` best checkpoint，使用新 canonical side rep 规则重新 analysis
  - `v27` best checkpoint，使用同一 analysis 规则
- 第一版只要求单 seed 先跑通，不在这一轮引入多 seed 比较

因此本轮真正的验收阈值写成：

- `side_from_side_rep(v27) > side_from_free_rep(v27)`
- `dataset_from_side_rep(v27) < dataset_from_side_rep(v26_reanalyzed)`
- `val_recon(v27) - val_recon(v26) <= 0.01`
- `val_side_group(v27) - val_side_group(v26) <= 0.10`

### 5.2 Canonical side rep 的 source of truth

这轮必须明确 canonical side rep 的来源，避免比较口径漂移。

规则如下：

- source of truth 是逐帧 `side_path_representation`
- canonical group-level side rep 统一由 analysis 端通过
  - `masked_mean_per_sequence(side_path_representation, valid_mask)`
  - 重新计算得到
- 不依赖 checkpoint 中是否已经缓存了某个 `group_pooled_side_rep`

也就是说：

- `v26_reanalyzed` 必须走 analysis 端重算
- `v27` analysis 也走同一套重算逻辑

这样可以保证基线和新 run 的 side rep 定义完全一致。

---

## 6. 配置与基线

第一版不需要新增很多新开关。

推荐做法：

- 保持现有框架默认值不变，避免影响历史脚本
- 新行为只进入新的 `v27` 实验线
- 不修改 `v26` preset 的既有训练语义

`v27` preset 中显式设置：

- `side_z_dim=8`

如果后续发现这个设置稳定有效，再考虑把它提升为新 baseline 默认值。

对应的新 run 可以命名为：

- `v27_side_basis_rep_tight_probe`

---

## 7. 不包含内容

这一轮明确不包含：

- side branch dataset adversarial
- private branch dataset classifier 重新引入
- residualized side branch
- attention pooling
- side basis 数量调整

原因很简单：

- 当前要验证的是“显式 basis rep + 小 side latent”本身是否足够压低 dataset leakage
- 如果同时混入 adversarial，就无法判断收益来自哪里

---

## 8. 风险

主要风险有两类：

1. `side_z_dim=8` 后，side branch 容量不足，导致 `val_side_group` 明显变差
2. dataset 信息并不主要藏在 `side_z`，而是已经通过 side basis activation 本身编码

对应判断标准：

- 如果 `side_group` 明显恶化，而 `dataset_from_side_rep` 没显著下降，说明这轮压缩没有击中问题
- 如果 `side_group` 维持住，且 `dataset_from_side_rep` 下降，则说明方案 1 成立

---

## 9. 实施范围

第一版预计修改：

- `scripts/lq/model/distnet.py`
- `scripts/lq/analyze_checkpoint.py`
- 新增 `scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh`
- `RESEARCH_PROGRESS.md`

第一版不改 dataset、不改 quantizer、不改主训练循环结构。
