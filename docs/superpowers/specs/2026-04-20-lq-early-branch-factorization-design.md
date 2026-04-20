# LQ Early Branch Factorization Design

**状态**: 草稿

**日期**: 2026-04-20

---

## 1. 背景

当前 `scripts/lq` 的 shared/side 因子化主要发生在量化之后：

- `feats -> avg_pool -> shared_head -> quantizer -> shared_quantized`
- 再从 `shared_quantized` 中硬切出 `side_latent` 与 `free_latent`

最近几轮实验已经给出稳定信号：

- `v23` 仍然是当前更好的 baseline
- `v24` 的 adversarial 只能部分改善 probe，不能解决根因
- `v25` 的 frame-wise QR 会改变下游读出几何，但不能阻止 raw latent 在上游已经纠缠
- `raw_linear_r2_free_to_side ≈ 1.0` 说明 side/free 在生成时几乎是同一份信息的不同重参数化

因此当前的主要矛盾不是“切分后约束不够强”，而是“切分发生得太晚”。

---

## 2. 目标

本设计希望通过更早的结构分流，让 `free` 与 `side` 从 feature map 阶段开始形成不同的信息路径。

第一版目标：

1. 不再从 `shared_quantized` 中切出 `side/free`
2. 让 `free` 分支独立承担 FSQ + shared basis reconstruction
3. 让 `side` 分支独立承担 side basis / side supervision
4. 保持 `private` 分支继续承担 residual / nuisance
5. 用最小结构改动验证：
   - `side_from_side_rep > side_from_free_rep`
   - `dataset_from_side_rep` 不高于 `v23`
   - latent raw coupling 明显下降
   - recon 不出现明显退化

第一版不追求：

- attention pooling
- side residual branch
- 再次引入 dataset auxiliary 主监督
- 更复杂的 token router / multi-token transformer 结构

---

## 3. 核心方案

### 3.1 结构改动

把当前：

- `encoder trunk -> global pooled shared bottleneck -> quantize -> split`

改为：

- `encoder trunk -> branch-specific adapters -> branch-specific pooling -> branch-specific heads`

新的数据流如下：

```text
x
-> encoder trunk
-> feats

feats -> free_adapter    -> free_pool    -> free_head    -> free_z    -> FSQ -> shared_free_recon
feats -> side_adapter    -> side_pool    -> side_head    -> side_z    -> side basis / side logits
feats -> private adapter -> private_pool -> private_head -> private_z -> private residual

final recon = shared_free_recon + shared_side_recon + private_weight * private_residual
```

### 3.2 设计原则

- `free` 与 `side` 不再共享同一个 pooled vector 入口
- `free` 是唯一进入 quantizer 的共享运动通道
- `side` 不再依赖 `shared_quantized` 的切片
- `private` 第一版尽量少动，避免一次修改三个归因点

---

## 4. 模块拆分

### 4.1 Encoder trunk

保留当前 `build_motion_encoder()` 输出：

- `initial_conv`
- `layer1`
- `layer2`
- `layer3`

`layer3` 的输出 `feats` 作为三路分支的共同输入。

### 4.2 Branch adapters

在 `feats` 后新增轻量 adapter：

- `free_adapter`
- `side_adapter`
- `private_adapter`

第一版建议使用轻量卷积块，例如：

- `3x3 conv + BN + ReLU`

目标不是增强容量，而是让三路在空间特征层上先分家。

第一版推荐默认配置：

- adapter 输出通道数保持为 `hidden_dim`
- 不单独扩宽 adapter 通道
- 先验证“更早分流”本身是否有效，而不是通过加宽容量掩盖问题

### 4.3 Branch pooling

每个分支独立池化，而不是共享一个 `avg_pool`。

第一版建议：

- `free_pool_size = 2`
- `side_pool_size = 2`
- `private_pool_size = 1` 或保持与当前兼容

优先使用 `AdaptiveAvgPool2d`，先不引入 attention pooling，避免额外变量。

### 4.4 Branch heads

- `free_head`: 输出 `free_z`
- `side_head`: 输出 `side_z`
- `private_head`: 输出 `private_z`

其中：

- `free_z` 接 quantizer、shared coeff heads、shared basis heads
- `side_z` 接 side semantic coeff/basis heads 与 side classifier
- `private_z` 接 residual decoder

当前 `shared_head -> quantizer -> split_side_free` 这套路径应整体退役。

第一版推荐默认维度：

- `free_z_dim = hidden_dim`
- `side_z_dim = hidden_dim`
- `private_z_dim = private_dim`

也就是说：

- `free_head` 与 `side_head` 默认都输出 `32`
- `private_head` 保持当前 `private_dim=32`

第一版不再人为把 `side/free` 压到更小子空间，先观察“早分流”本身能否打破高度线性耦合。

---

## 5. Forward 逻辑

第一版 forward 应按下列顺序组织：

1. `x -> feats`
2. `feats -> free_feats / side_feats / private_feats`
3. 分别池化并 flatten
4. 得到 `free_z / side_z / private_z`
5. 只对 `free_z` 做 quantization
6. 用 quantized `free_z` 做 shared free reconstruction
7. 用 `side_z` 做 side reconstruction 与 side supervision
8. 用 `private_z` 做 residual reconstruction
9. 三路相加得到最终输出

第一版需要显式删除或停用这些旧逻辑：

- 从 `shared_quantized` 切出的 `side_latent_raw/free_latent_raw`
- `side_free_frame_qr`
- 基于 “同一 quantized latent 切片” 的 side/free 正交化路径

相关字段可以暂时保留在 config 中做向后兼容，但新 baseline 默认不再启用。

---

## 6. 损失与分析

### 6.1 损失

第一版尽量保守：

- 保留 `recon`
- 保留 `shared_recon`
- 保留 `lq`
- 保留 `orth`
- 保留 `basis_l1`
- 保留 `residual`
- 保留 `side_group`

第一版默认不再把以下项作为主驱动：

- `subspace_orth`
- `free_side_adv`
- `side_free_frame_qr`

如果相关代码保留，默认应设为关闭，仅作为 ablation 开关。

其中 `orth` 在新结构中的定义需要明确：

- `orth` 仍然只约束 `shared free basis bank`
- 不把 `side_z/free_z` 的正交性重新塞回主损失
- `side basis bank` 若继续沿用现有矩阵约束，可单独保留其结构化约束，但不与 shared basis 做联合 QR

这样可以避免“basis 正交”和“latent 解耦”再次混成一个目标。

### 6.2 分析

分析脚本需要切换到新结构的表示导出：

- `group_pooled_side_rep`
- `group_pooled_free_rep`
- `group_pooled_side_latent`
- `group_pooled_free_latent`

并继续做：

- `side_from_side_rep`
- `side_from_free_rep`
- `dataset_from_side_rep`
- `dataset_from_free_rep`
- `dataset_from_private_rep`
- side/free latent 线性 recoverability

新的 latent recoverability 应针对真正独立生成的 `side_z/free_z` 计算，而不是旧切片。

---

## 7. 成功标准

第一版是否成功，以相对 `v23` 的结构性改善为主，不以单一总 loss 为唯一标准。

优先判断顺序：

1. `side_from_side_rep > side_from_free_rep`
2. `dataset_from_side_rep <= v23`
3. `raw_linear_r2_free_to_side < 0.95`
4. `val_recon - v23_val_recon <= 0.02`

如果 probe 改善但 `recon` 略有上升，可以接受。

如果 `recon` 很好但 side/free probe 继续反向或高度对称，则视为失败。

---

## 8. 风险

主要风险有三类：

1. `free` 分支容量不足，shared reconstruction 明显下滑
2. `side` 分支过弱，只剩监督头在读 dataset/shortcut
3. `private` 继续吸走大部分解释压力，导致 shared/side 都变弱

对应的第一层应对策略：

- 优先调 `pool_size`
- 其次调 adapter 通道数
- 最后再考虑 residual-side 或更强监督

不建议在第一版一开始就重新引入大量 adversarial / QR / orth trick，否则无法判断结构改动是否真的有效。

---

## 9. 实施范围

第一版预计修改这些文件：

- `scripts/lq/model/distnet.py`
- `scripts/lq/model/heads.py`
- `scripts/lq/model/encoder.py`
- `scripts/lq/train.py`
- `scripts/lq/training/config.py`
- `scripts/lq/analyze_checkpoint.py`

第一版不要求重写 dataset、train loop 或分析框架，只要求把表示来源切换到新的 branch 架构。
