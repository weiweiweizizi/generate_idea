# DisentangleNet Training Observations

> 记录 disentangleNet v31 训练过程中的 Loss 趋势观察和消融实验发现。
> 最后更新：2026-05-08

---

## 1. v31 Baseline Loss 分解（X vs Y）

### 训练配置

| 参数 | 值 |
|------|-----|
| model | DistNet v31 |
| levels | (2, 6)，共 8 个 action basis |
| basis_size | 119 × 119（mouth 区域） |
| private_dim | 32 |
| private_residual_weight | 0.05 |
| side_group_loss_weight | 0.3 |
| lq_weight | 10.0（但 lq 全程 = 0，FSQ 未激活） |
| orth_weight | 0.1 |
| basis_l1_weight | 1.0 |
| residual_weight | 0.02 |

### 50 Epochs 结果对比

| 分量 | X Ep1 | X Ep50 | X Δ | Y Ep1 | Y Ep50 | Y Δ |
|------|-------|--------|-----|-------|--------|-----|
| **val_loss** | 1.069 | 0.987 | -7.7% | 1.059 | 0.988 | -6.7% |
| **val recon** | 0.364 | 0.350 | -3.7% | 0.359 | 0.350 | -2.6% |
| **val residual** | 0.047 | **0.319** | **+582%** | 0.047 | 0.016 | -66% |
| **val side_group** | 1.097 | 0.881 | -20% | 1.096 | 0.943 | -14% |
| **val side_loss** | 1.093 | 1.440 | +32% | 1.093 | 1.294 | +18% |
| **val basis_l1** | 0.012 | 0.004 | -62% | 0.012 | 0.004 | -63% |

### 关键发现

1. **`lq` 全程 = 0**：FSQ 量化通道完全没有激活，latent quantization 形同虚设。
2. **`orth ≈ 0`**：QR 正交化强制满足，orthogonalization penalty 无需工作。
3. **`basis_l1` 健康**：train/val 对齐良好，基稀疏性正则正常。
4. **X 的 residual 严重过拟合**：从 0.047 飙升到 0.319（+582%），private decoder 在背训练集。
5. **side_group 严重过拟合**：两方向 train/val gap 均 ~0.6，side 监督信号与重建竞争梯度。
6. **side_loss 上升**：训练后期 side_loss 不降反升，说明 `side_group` 压下去了但分类任务本身反而被干扰。

---

## 2. Pure Reconstruction Tuning 实验

### 改动

| 参数 | v31 | Pure Recon |
|------|-----|------------|
| side_group_loss_weight | 0.3 | 0.01（等效关闭） |
| private_residual_weight | 0.05 | **0.20** |
| private_dim | 32 | **64** |
| lq_weight | 10.0 | 0.0 |
| orth_weight | 0.1 | 0.0 |
| basis_l1_weight | 1.0 | 0.0 |
| residual_weight | 0.02 | 0.0 |

### 结果对比

| | v31 X | Pure X | 改进 | v31 Y | Pure Y | 改进 |
|---|---|---|---|---|---|---|
| **val_loss** | 0.968 | **0.484** | **-50.0%** | 0.971 | **0.462** | **-52.4%** |
| **val recon** | 0.350 | **0.237** | **-32.4%** | 0.350 | **-0.222** | **-36.6%** |
| **val residual** | 0.319 | 0.080 | -75% | 0.016 | 0.170 | +10x |
| Best epoch | ep45 | ep49 | | ep48 | ep50 | |

### 结论

关闭 side 干扰后：
- **recon 降低了 32-37%**，说明 side_group_weight=0.3 严重干扰了重建梯度
- **过拟合消失**：train/val gap 从 ~0.17 降到 ~0.02
- Y 的 private decoder 正常激活（从 0.016 升到 0.170），X 的 private 最终被放弃（从 0.319 降到 0.080）

---

## 3. Private Residual 驼峰现象（核心发现）

### 现象描述

在 Pure Recon 实验中，X 和 Y 两方向的 private residual 都出现了一个 **"驼峰"**：

```
Private Residual 变化轨迹（Train）:

    0.80 |              ★ X峰值(ep24)
         |           ●
    0.70 |         ●
         |       ●        ★ Y峰值(ep23)
    0.60 |     ●        ●
         |   ●        ●
    0.40 | ●      ●
         |      ●
    0.20 |   ●
         | ●
    0.05 |●
         +---+---+---+---+---+---+---+---+---→ epoch
           1   5   10  15  20  25  30  35  40  45  50
```

### 分阶段数据

#### X 方向

| Epoch | t_shared | t_recon | t_residual | 含义 |
|-------|----------|---------|------------|------|
| 1 | 0.3625 | 0.3626 | 0.051 | 初始化 |
| 15 | 0.358 | 0.288 | 0.60 | ⚠️ private 过激活 |
| **22** | **0.336** | **0.256** | **0.80** | **⚠️ 驼峰峰值：private 帮大忙** |
| 25 | 0.292 | 0.235 | 0.72 | private 开始收缩 |
| 30 | 0.264 | 0.229 | 0.41 | shared 接管 |
| 50 | 0.215 | 0.214 | 0.08 | 收敛稳定 |

#### Y 方向

| Epoch | t_shared | t_recon | t_residual | 含义 |
|-------|----------|---------|------------|------|
| 1 | 0.3593 | 0.3594 | 0.050 | 初始化 |
| 15 | 0.356 | 0.289 | 0.62 | ⚠️ private 过激活 |
| **23** | **0.328** | **0.247** | **0.81** | **⚠️ 驼峰峰值** |
| 30 | 0.262 | 0.229 | 0.40 | shared 追赶 |
| 50 | 0.212 | 0.207 | 0.18 | 收敛（保留了 private） |

### 数学解释

驼峰期间，重建由两项构成：

```
recon = shared_reconstruction - private_residual_weight × |private_residual|
       = shared_recon          - 0.2 × |private_residual|

驼峰峰值时：recon < shared_recon
  X ep22: 0.256 < 0.336，gap = -0.080
  Y ep23: 0.247 < 0.328，gap = -0.081
```

这意味着 private decoder 在驼峰期间**实际上在帮倒忙修正重建误差**（用减法修正 shared 重建过头的部分）。但由于没有任何正则约束，private 学得过大（|residual| ~ 0.8），在总 loss 中占主导，导致 shared basis 失去了学习压力。

峰值后：private residual 收缩 → shared basis 重新独立学习 → 最终收敛。

### X vs Y 的差异

| | X | Y |
|---|---|---|
| 驼峰峰值 epoch | ep24 | ep23 |
| 峰值 residual | 0.80 | 0.81 |
| **最终 residual** | **0.08**（几乎放弃） | **0.18**（保留 ~45%） |
| shared 最终主导程度 | shared 几乎 100% | shared ~75% + private ~25% |
| shared 追赶时机 | ep25 后急剧 | ep25 后缓慢 |

**Y 的 private decoder 最终保留了 ~45% 的重建贡献**，这使得 Y 的最终 val_loss（0.462）优于 X（0.484）。

### 根因假设

Private 驼峰的根本原因是 **shared basis 的学习速度远慢于 private decoder**：

1. shared reconstruction 依赖 8 个 action basis 的线性组合 + 标量系数
2. private decoder 是一个独立的 MLP，可以直接拟合残差
3. private 的学习自由度远高于 shared，因此学得更快、更大
4. 在没有正则的情况下，private 淹没了 shared 的学习信号

---

## 4. 下一步消融计划

### 高优先级

- [ ] **增加 action basis 数量**：`levels=(2,3,12)` 或 `(2,3,15)`，增大 shared 容量，看驼峰是否消失
- [x] **Private residual weight cosine schedule**：从 0.2 开始，在 ep15-30 逐渐降到 0.02，抑制驼峰峰值（见 §5）
- [x] **Shared coeff head 学习率放大**：3× base lr，让 shared 在前期就快速学习（见 §5）

### 中优先级

- [ ] **纯重建（无 private）**：设置 `private_residual_weight=0`，看纯 shared 能做到多少 recon
- [ ] **Y 方向增大 private_dim**：Y 的 private 在有效工作，可试 128
- [ ] **FSQ 量化修复**：排查为什么 lq=0，修复后重新训练

### 低优先级

- [ ] **侧别分离实验**：在不同侧别（left_affected, bilateral_normal, right_affected）子集上分别训练，看 basis 的侧别路由是否稳定
- [ ] **跨数据集泛化**：在 IMR 训练，在 TT 上验证，测试 shared basis 的跨数据集能力

---

## 5. v2 实验：Cosine Schedule + Shared LR 放大

### 改动

| 参数 | v1 | v2 |
|------|-----|-----|
| Epochs | 50 | 100 |
| `private_residual_weight` | 固定 0.20 | **Cosine schedule**：ep1-15=0.20，ep15-30→0.02，ep30+=0.02 |
| shared_coeff_heads/net LR | 同等 3e-4 | **3× = 9e-4** |
| 其他参数 LR | 3e-4 | 3e-4 |
| private_dim | 64 | 64 |

### 结果对比

#### X 方向

| | v1 (50ep) | v2 (100ep) | 变化 |
|---|---|---|---|
| **val_recon** | 0.237 | **0.188** | **-20.7% ✅** |
| **val_loss** | 0.484 | **0.388** | **-19.8% ✅** |
| val_residual | 0.080 | 0.599 | private 不再被放弃，持续参与 |
| train/val gap | 0.023 | **0.007** | **-69.6% ✅** |
| 驼峰峰值 | ~0.80 (ep24) | ~0.86 (ep30) | 峰值略高但未崩塌 |

#### Y 方向

| | v1 (50ep) | v2 (100ep) | 变化 |
|---|---|---|---|
| **val_recon** | 0.223 | **0.209** | **-6.3% ✅** |
| **val_loss** | 0.463 | **0.428** | **-7.6% ✅** |
| val_residual | 0.170 | 0.742 | private 持续参与 |
| train/val gap | 0.016 | **0.014** | **-12.5% ✅** |

### 核心发现

1. **Cosine schedule 抑制了"先飙升再放弃"的驼峰崩溃**：v1 里 X 的 private 在 ep24 冲到 0.80 后迅速被放弃（ep50 → 0.08）；v2 里 private 在 ep30 冲到 ~0.86 后平稳保持在 ~0.60，**不再崩溃**。

2. **Shared 3× LR 让 shared basis 真正接管重建**：v1 里 X 的 shared 在 ep1-10 几乎不动，靠 private 撑场；v2 里 shared 全程学习，val_recon 额外压低了 0.05。

3. **Train/val gap 显著缩小**：X 的 gap 从 0.023 降到 0.007，过拟合几乎消失。

4. **Y 方向提升幅度小于 X**：Y 的 shared LR 放大效果不如 X 明显，说明 Y 的 shared 本身学习效率已经较高。

### 完整版本对比（汇总）

| 版本 | X val_loss | Y val_loss | X val_recon | Y val_recon |
|------|-----------|-----------|------------|------------|
| v31 baseline (50ep) | 0.968 | 0.971 | 0.350 | 0.350 |
| v1 pure_recon (50ep) | 0.484 | 0.463 | 0.237 | 0.223 |
| **v2 cosine+3xLR (100ep)** | **0.388** | **0.428** | **0.188** | **0.209** |

### 下一步

- [x] **纯 private=0**：关掉 private 分支，看纯 shared 能做到多少 recon → 见 §7
- [x] **增加 action basis**：levels=(3,4,4)，11 shared bases → 见 §7
- [ ] **FSQ 修复**：排查 latent quantization 梯度消失问题
- [ ] **Y private_dim=128**：Y 的 private 在 v2 中持续参与，可尝试增大容量

---

## 6. v3 实验：Pure Shared（private=0）+ 增加 Action Basis

### 实验设计

| | ExpA | ExpB |
|-----|------|------|
| levels | `(2,6)` | `(3,4,4)` |
| shared bases | **8**（8-slot bank） | **11**（15-slot bank 裁剪前 11） |
| side bases | 3（from level2） | 3（from 15-slot bank 裁剪前 3） |
| private | **0（完全关闭）** | **0（完全关闭）** |
| 其他 | 同 v2 | 同 v2 |

### 核心问题

> 如果纯 shared（无 private）能达到 v2 的 90%+ 水平，说明 private decoder 的贡献极小，模型结构可以大幅简化。

### 结果对比

#### X 方向

| Epoch | v2 val_recon | ExpA val_recon | ExpB val_recon |
|-------|-------------|--------------|--------------|
| 1 | 0.364 | 0.364 | 0.364 |
| 15 | — | 0.344 | 0.364 |
| 30 | — | 0.255 | 0.333 |
| 50 | — | 0.236 | 0.279 |
| 70 | — | 0.235 | 0.255 |
| 100 | 0.188 | 0.212 | 0.242 |
| **Best** | **0.188** | **0.212**（ep100） | **0.242**（ep90） |

#### Y 方向

| Epoch | v2 val_recon | ExpA val_recon | ExpB val_recon |
|-------|-------------|--------------|--------------|
| 1 | 0.359 | 0.359 | 0.359 |
| 15 | — | 0.345 | 0.347 |
| 30 | — | 0.265 | 0.314 |
| 50 | — | 0.232 | 0.251 |
| 70 | — | 0.215 | 0.217 |
| 100 | 0.209 | 0.211 | 0.213 |
| **Best** | **0.209** | **0.211**（ep90） | **0.213**（ep80） |

### 汇总表

| 版本 | X val_recon | Y val_recon | X val_loss | Y val_loss | 备注 |
|------|------------|------------|------------|------------|------|
| v31 baseline (50ep) | 0.350 | 0.350 | 0.968 | 0.971 | side=0.3 干扰 |
| v1 pure_recon (50ep) | 0.237 | 0.223 | 0.484 | 0.463 | 有private=0.2 |
| v2 cosine+3xLR (100ep) | 0.188 | 0.209 | 0.388 | 0.428 | 有private+cschedule |
| **ExpA 8bases private=0** | **0.212** | **0.211** | **0.472** | **0.433** | 纯shared，无驼峰 |
| **ExpB 11bases private=0** | **0.242** | **0.213** | **0.512** | **0.435** | 更多bases，反而略差 |

### 关键发现

1. **纯 shared 已经非常接近 v2（有 private）水平**：
   - Y 方向：ExpA=0.211 vs v2=0.209，差 < 1%
   - X 方向：ExpA=0.212 vs v2=0.188，差 ~13%（private 在 X 方向有贡献）

2. **11 个 bases 反而不如 8 个 bases（ExpB > ExpA）**：
   - X 方向：ExpB=0.242 vs ExpA=0.212
   - 可能原因：15-slot bank 的 basis 质量不如专门的 8-slot bank（v1/v2 用的是 8-slot）
   - 或者：basis 数量增加后，coefficient 学习困难，需要更多正则

3. **无驼峰现象**：ExpA/B 的 residual 全程稳定在 0.047-0.051，没有 v1/v2 里的先飙升后崩塌。

4. **Y 方向更稳定**：Y 的 ExpA 和 ExpB 几乎相同（0.211 vs 0.213），说明 Y 的 shared 本身学习效率已经很高。

### 下一步

- [ ] **用高质量 11-slot basis 重新跑 ExpB**：用类似 8-slot 的生成方式（cluster_count 对应 levels=3,4,4），而不是从 15-slot 裁剪
- [ ] **Private 对 X 的贡献拆解**：X v2 比 ExpA 好 13%，需要搞清楚 private 在 X 方向具体做了什么
- [ ] **增大 ExpA 的 epochs**：ExpA Y ep90 后还在下降，可延长到 200 epochs
- [ ] **FSQ 修复**：排查 latent quantization 梯度消失问题

---

## 8. v4 实验：从 Checkpoint 热启 + Side Supervision

### 实验设计

| | v4 X | v4 Y |
|-----|------|------|
| 起点 | v2_recon_x/best.pt (ep100, val_loss=0.388) | v3_expA_pureShared8_y/best.pt (ep90, val_loss=0.433) |
| private | **保留**（cosine schedule, 0.20→0.02） | **关闭**（private=0） |
| side_group schedule | ep1-20=0.01, ep20-50=0.05, ep50-100=0.10 | 同左 |
| action/side basis | **不 freeze**，继续训练 | 同左 |
| 继续训练 | 100 epochs | 100 epochs |

### 结果

#### v4 X

| Epoch | side_weight | val_recon | val_loss | val_sg | 备注 |
|-------|------------|-----------|---------|--------|------|
| 1 | 0.01 | 0.201 | 0.421 | 1.091 | 起点 |
| 20 | 0.01 | 0.186 | **0.383** | 1.094 | **最佳** |
| 30 | 0.05 | 0.186 | 0.427 | 1.093 | side_weight 增大后 loss 上升 |
| 60 | 0.10 | 0.190 | 0.488 | 1.076 | 继续恶化 |
| 90 | 0.10 | 0.182 | 0.453 | 0.882 | side_group 下降但 val 仍差 |

**关键发现**：side_weight 增大后 loss 反而上升，说明当前 side supervision 信号与重建竞争。最佳结果就是 ep20 的 0.383，略好于 v2 的 0.388。

#### v4 Y

| Epoch | side_weight | val_recon | val_loss | val_sg | 备注 |
|-------|------------|-----------|---------|--------|------|
| 1 | 0.01 | 0.218 | 0.448 | 1.161 | 起点 |
| 5 | 0.01 | 0.210 | **0.432** | 1.159 | **最佳（和 ExpA 一样）** |
| 20 | 0.01 | 0.210 | 0.433 | 1.157 | |
| 50 | 0.05 | 0.212 | 0.486 | 1.243 | val_sg 开始过拟合上升 |
| 80 | 0.10 | 0.224 | 0.619 | 1.705 | 严重过拟合 |

**关键发现**：添加 side supervision 后 Y 的 val_sg 持续上升（1.161→1.705），严重过拟合。最佳结果仍是 checkpoint 起点（0.432），没有任何提升。

### 完整版本汇总

| 版本 | X val_recon | Y val_recon | X val_loss | Y val_loss |
|------|------------|------------|------------|------------|
| v31 baseline | 0.350 | 0.350 | 0.968 | 0.971 |
| v2 cosine+3xLR | 0.188 | 0.209 | 0.388 | 0.428 |
| ExpA (8bases, private=0) | 0.212 | 0.211 | 0.472 | 0.433 |
| ExpB (11bases, private=0) | 0.242 | 0.213 | 0.512 | 0.435 |
| **v4 X (v2 + side sched)** | **0.186** | — | **0.383** | — |
| **v4 Y (ExpA + side sched)** | — | **0.210** | — | **0.431**（无提升） |

### 核心结论

1. **X 方向**：v4 在 ep20 达到 val_loss=0.383，比 v2 略有改善（0.388→0.383），private + side 组合有效
2. **Y 方向**：side supervision 完全无效，val_loss 从 0.432 恶化到 0.619。ExpA 的 Y 已经到顶，side 监督是干扰
3. **Side supervision 过拟合**：v4 Y 的 val_sg 从 1.16 飙升到 1.71，说明 side_group loss 压低了训练 loss 但 val loss 上升，side 路由在背训练集

### 下一步

- [ ] **X 方向单独强化 private**：搞清楚 private 在 X 方向具体在做什么（静态结构修正？跨帧平滑？）
- [ ] **Y 方向不要再加 side supervision**：ExpA Y 已经是最优，资源投向 X
- [ ] **X 方向从 v4 checkpoint 继续**：在 0.383 的基础上降低 side_weight 或调整 private schedule
- [ ] **FSQ 修复**：排查 latent quantization 梯度消失问题

---

## 8. v5 实验：冻结 private_decoder + action_basis_bank，只训 side_branch

从 v2_recon_x/best.pt 热启：
- **frozen**: private_decoder（1.0M）+ action_basis_bank（1.0M）
- **train**: side_branch（side_basis_bank, side_adapter, side_head, side_coeff_*）+ shared_coeff_heads/net
- side_group_schedule: ep1-20=0.01, ep20-50=0.05, ep50-100=0.10, private cosine schedule 同 v2

### 结果

| Epoch | side_w | pr_w | val_loss | val_recon | val_sg | 备注 |
|-------|--------|------|----------|-----------|--------|------|
| 1 | 0.01 | 0.20 | 0.427 | 0.206 | 1.091 | 起点 |
| 15 | 0.01 | 0.20 | **0.387** | 0.188 | 1.097 | |
| **20** | **0.01** | **0.155** | **0.384** | **0.186** | **1.096** | **最佳** |
| 25 | 0.05 | 0.065 | 0.426 | 0.185 | 1.098 | side↑ 后恶化 |
| 60 | 0.10 | 0.02 | 0.466 | 0.183 | 0.992 | side_group↓ 但 recon 恶化 |
| 90 | 0.10 | 0.02 | 0.464 | 0.183 | 0.970 | |

### 对比

| 版本 | X val_loss | 变化 |
|------|-----------|------|
| v2（cosine+3xLR） | 0.388 | baseline |
| v4（v2 + side sched，全训） | 0.383 | -1.3% |
| **v5（freeze private+action，只训side）** | **0.384** | -1.0% |

### 核心结论

1. **冻结 private + action 后，side_branch 可以独立优化**：v5 在 ep20 达到 0.384，与 v4（0.383）几乎相同
2. **Private freeze 不影响最终结果**：v5 和 v4 的最佳点都在 ep20 左右，且数值接近，说明 private_decoder 的作用是在训练初期帮助重建，而 ep20 后它贡献稳定，不需要继续训练
3. **Side_group 权重增大后 loss 仍上升**：side_w 从 0.01→0.05→0.10 后，val_loss 持续恶化，说明 side supervision 的价值有限，可能已经饱和

### 下一步

- [ ] **X 方向从 v5 checkpoint 继续（freeze 不变，side_weight 保持 0.01）**：看更长的 schedule 能否进一步降低 recon
- [ ] **对比 freeze vs unfreeze 的 side_branch 效果**：v4 vs v5 的 side_sg 值是否不同
- [ ] **Private 在 X 方向的定量贡献**：用不同 private_dim 或不同 freeze 策略做 ablation

---

## 9. 验证：Action Coefficients 能预测 Side Labels

从 v2_recon_x/best.pt 加载模型，提取 `free_path_coefficients`（mean pool over frames → shape `(N, 2)`），
用 sklearn LogisticRegression 在 train 上训练，val 上测试。

### 结果

**Action coefficients（2维）→ Side label（3分类）**

| 模型 | Val Accuracy | vs Random |
|------|-------------|-----------|
| Baseline（随机） | 33.3% | — |
| **LR on action coefficients** | **80.7%** | **+142%** |
| Level 0 单独 | 36.8% | ≈随机 |
| Level 1 单独 | 78.9% | +137% |

Per-class val 报告：
```
              precision  recall  f1-score  support
class 0 (left_affected)     0.812   0.765    0.788       17
class 1 (bilateral_normal)  0.778   0.700    0.737       20
class 2 (right_affected)    0.826   0.950    0.884       20
```

### 核心发现

1. **Action coefficients 携带极强的侧别信息**：仅用 2 个标量（mean pooled coefficients），logistic regression 达到 80.7%，说明 action branch 的 basis usage 模式已经隐式编码了侧别
2. **Level 1 是侧别主成分**：Level 0 ≈ 随机，Level 1 单独就能到 79%，说明侧别信号集中在某个 level 的 basis slot 上
3. **这解释了为什么 side_branch 学不到东西**：side_branch 走的是独立 encoder path，它的 latent 跟 action branch 完全不共享信息，没有利用到 action coefficients 里已有的侧别信号

### 下一步

- [ ] **在 action coefficients 上加 side head**：从 `free_path_coefficients`（经 mean pool）接 MLP 预测 side，辅助 loss 直接加在 action branch 上
- [ ] **Y 方向同样验证**：从 ExpA Y 的 checkpoint 做同样实验，看 Y 方向的 action coefficients 是否也有侧别可分性

---

## 10. 已知问题

1. **FSQ 未激活**：`lq_weight=10.0` 但 `lq=0`，需要排查 FSQ 梯度是否被正确传递
2. **Config 校验过严**：`prepare_train_config` 要求 `group_side_loss_weight > 0`，限制了消融实验设计
3. **Side loss 和 side_group 相互干扰**：训练后期 side_loss 上升，与 side_group 的下降形成对抗
