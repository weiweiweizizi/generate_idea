# Research Idea Report

[TOC]

**Direction**: 基于 `diff of distance matrix` 的面部子运动分解与后验运动解释  
**Current focus**: `scripts/lq` 前置探索，`scripts/disentangleNet` 冻结主线，`scripts/matrix_vis` 后验可视化  
**Last Updated**: 2026-04-29

---

## 1. 当前问题定义

当前研究已经从“是否存在可行想法”进入“哪条结构主线最可信”的阶段。

更准确的研究目标可以表述为：

1. 从窗口级面部距离差矩阵中学习跨被试共享的运动基
2. 将侧别信息尽量路由到显式 side branch，而不是混在 free/shared 中
3. 控制 private residual，避免模型把解释压力都推回私有分支
4. 用后验轨迹重建工具把 basis / observation 的语义落到可视轨迹上

当前不应再把项目描述成“同时平行探索 NMF、Tucker、CP、深度网络等多个候选 idea”。从代码与结果看，主线已经实质收敛到：

- `scripts/lq`
  - 结构探索与消融
- `scripts/disentangleNet`
  - 接受版 `v31`
- `scripts/matrix_vis`
  - basis / observation 的后验解释层

---

## 2. 数据与实验范围

当前主线使用的数据与设置：

| 项目 | 当前主设置 |
|------|------------|
| data roots | `data/win20-step20/IMR,data/win20-step20/TT` |
| mode | `x` |
| region | `mouth` |
| input unit | grouped window sequence |
| group size | `4` |
| supervision focus | side-aware shared basis learning |

当前 full post-hoc probe 的统计范围：

- `293` groups
- `267` subjects
- dataset counts:
  - `IMR = 228`
  - `TT = 65`
- side counts:
  - `Left = 83`
  - `Normal = 111`
  - `Right = 99`

---

## 3. `scripts/lq`：前置探索的关键数据

当前 `lq_*` 运行目录已统一整理到：

- `outputs/lq/win20-step20/`

当前仓库实际保留的 `lq` 运行目录主要是 `v20-v32`：

| 分组 | 保留目录 |
|------|----------|
| private-cap tradeoff | `outputs/lq/win20-step20/lq_x_mouth_v20_residual_fsq_sparse_shared_privatecap04_probe_win20_e50` |
| private-cap tradeoff | `outputs/lq/win20-step20/lq_x_mouth_v21_residual_fsq_sparse_shared_privatecap06_probe_win20_e50` |
| side semantic bank | `outputs/lq/win20-step20/lq_x_mouth_v22_side_semantic_bank_probe_win20_e50` |
| side semantic bank | `outputs/lq/win20-step20/lq_x_mouth_v23_side_subspace_orth_probe_win20_e50` |
| side route probes | `outputs/lq/win20-step20/lq_x_mouth_v24_free_side_adv_probe_win20_e50` |
| side route probes | `outputs/lq/win20-step20/lq_x_mouth_v25_frame_qr_probe_win20_e50` |
| side route probes | `outputs/lq/win20-step20/lq_x_mouth_v26_early_branch_probe_win20_e50` |
| side route probes | `outputs/lq/win20-step20/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50` |
| side route probes | `outputs/lq/win20-step20/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50` |
| laterality / joint-QR | `outputs/lq/win20-step20/lq_x_mouth_v29_laterality_contrast_probe_win20_e50` |
| laterality / joint-QR | `outputs/lq/win20-step20/lq_x_mouth_v30_joint_qr_levels26_side3_probe_win20_e50` |
| laterality / joint-QR | `outputs/lq/win20-step20/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50` |
| severity branch | `outputs/lq/win20-step20/lq_x_mouth_v32_joint_qr_levels26_side3_sparse_side_severity_probe_win20_e50` |

更早期的 `v10-v19` 在当前仓库里主要以文档结论保留，不再作为 `outputs/` 顶层运行目录索引。

## 3.1 Collapse 被真正缓解的起点是 `v10`

`v10` 是第一个可以被当作可信 baseline 的版本。

关键指标：

| 指标 | 数值 |
|------|------|
| `val_loss` | `0.3619` |
| `val_recon` | `0.3600` |
| `val_shared_recon` | `0.3620` |
| `val_scaled_residual` | `0.0034` |
| L2 usage | `[20, 23, 37]` |
| L3 usage | `[18, 2, 3, 3, 25, 29]` |

核心意义：

- 官方 `FSQ` 替换是早期 anti-collapse 的关键改动
- 从 `v10` 开始，问题不再是“code 完全不用”，而是“code 怎么用才更可解释”

## 3.2 `v17-v21` 确立了 shared / private 的主要 tradeoff

这组版本决定了后面 side-aware 结构的出发点。

| 版本 | 结构变化 | `val_recon` | `val_shared_recon` | `val_scaled_residual` |
|------|----------|-------------|--------------------|-----------------------|
| `v17` | `residual_fsq + global_qr + basis_l1` | `0.3109` | `0.3423` | `0.0393` |
| `v18` | `v17 + sparse_shared_mixing + shared_recon_loss` | `0.3150` | `0.3469` | `0.0394` |
| `v19` | `v18 + private cap=0.5` | `0.3275` | `0.3446` | `0.0214` |
| `v20` | `v19 + private cap=0.4` | `0.3310` | `0.3448` | `0.0171` |
| `v21` | `v19 + private cap=0.6` | `0.3245` | `0.3446` | `0.0251` |

这组结果对应的判断：

1. `residual_fsq` 有助于 higher-level usage
2. 直接约束 basis 并不能自动得到更好的解释，private residual 会把收益拿走
3. `shared_recon` 必须单独监督
4. `v19` 是这一阶段最稳的 interpretability baseline

## 3.3 side-aware 路线的关键结果

### `v22`

- `val_recon = 0.3330`
- `val_shared_recon = 0.3504`
- `val_scaled_residual = 0.0216`
- `mean_side_path_usage = 0.500`
- `side_from_side_rep_acc = 0.4545`
- `dataset_from_side_rep_acc = 0.8182`

解释：

- side path 已经被用上
- 但 2 个 side basis 还不足以形成有效 separation

### `v23`

- `val_recon = 0.3342`
- `val_shared_recon = 0.3504`
- `val_scaled_residual = 0.0208`
- `side_from_side_rep_acc = 0.5227`
- `side_from_free_rep_acc = 0.5182`
- `raw_linear_r2_free_to_side ≈ 1.0`

解释：

- `subspace_orth` 并没有真正把 side / free 分开
- 两条分支几乎线性可逆，因此这条路被否定

### `v29-v31`

这一段的目标是把 laterality 信息稳定地压进 side branch。

`v29`：

- `side_from_usage_acc = 0.6773`
- `side_from_usage_coeff_acc = 0.7227`
- Left 平均偏向 `b0`
- Right 平均偏向 `b1`

`v30`：

- `side_from_usage_acc = 0.8318`
- `side_from_usage_coeff_acc = 0.8045`
- 3 个 side basis 的结构更简洁，也更接近后续冻结版

这组结果说明：

- laterality 确实可以被 side branch 显式承载
- 但 `scripts/lq` 里的 `v31` 仍应视为探索期版本，不是最终冻结主线

## 3.4 severity 结果仍弱

`v32` 的 severity interpretability 结果：

| 任务 | Accuracy | Balanced Acc | Macro F1 |
|------|----------|--------------|----------|
| `severity_from_level2_coeff` | `0.6212` | `0.5000` | `0.3832` |
| `severity_from_level2_rep` | `0.6621` | `0.5699` | `0.5377` |
| `severity_from_level2_usage` | `0.6621` | `0.5699` | `0.5377` |

当前只能保守写成：

- severity 在当前结构下是弱信号
- side-aware 中层表示对 severity 有有限相关
- 不能写成“severity 已成功解耦”

---

## 4. `scripts/disentangleNet`：当前接受主线的关键数据

## 4.1 固定配置

当前冻结在 `train.py` 里的 `v31` 主设置：

| 项目 | 值 |
|------|----|
| `levels` | `2,6` |
| `quantizer_type` | `residual_fsq` |
| `basis_orthogonalization` | `joint_global_qr` |
| `side_basis_count` | `3` |
| `side_pooling` | `fixed_region2_contrast` |
| `early_branch_factorization` | `True` |
| `private_residual_max_l1` | `0.5` |

这说明 `scripts/disentangleNet` 不是继续开放搜索空间，而是固定了一个已经接受的结构组合。

## 4.2 checkpoint 级指标

`v31_current_verify`：

| 指标 | 数值 |
|------|------|
| `val_recon` | `0.3097` |
| `val_shared_recon` | `0.3255` |
| `val_scaled_residual` | `0.0209` |

`v31_internal_compact_verify_e50`：

| 指标 | 数值 |
|------|------|
| `val_recon` | `0.3226` |
| `val_shared_recon` | `0.3377` |
| `val_scaled_residual` | `0.0199` |

当前可稳定复述的区间：

- `val_recon ≈ 0.31 - 0.32`
- `val_shared_recon ≈ 0.326 - 0.338`
- `val_scaled_residual ≈ 0.020`

## 4.3 Full k-fold probe

`293` groups / `267` subjects / `5` folds 下的关键结果：

| 任务 | Accuracy 范围 | 含义 |
|------|---------------|------|
| `side_from_side_rep` | `0.9283 - 0.9317` | side branch 强烈承载了侧别 |
| `side_from_usage_coeff` | `0.9283 - 0.9420` | side usage + coeff 几乎可直接判 side |
| `side_from_free_rep` | `0.4710` | free branch 的 side 显式信息明显弱很多 |
| `dataset_from_side_rep` | `0.7713 - 0.7782` | side branch 仍保留 dataset 痕迹 |
| `dataset_from_free_rep` | `0.7986 - 0.8055` | free branch 的 dataset 痕迹更明显 |
| `dataset_from_private_rep` | `0.8840 - 0.8908` | private branch 是最强 dataset carrier |

这里最重要的结论不是“acc 高”，而是信息分布：

1. side 信息已经被 side branch 稳定承载
2. free / private 还没有摆脱 dataset leakage
3. private branch 仍是 dataset 偏差的主通道

## 4.4 code usage 的保留问题

当前 `v31` 的 free quantizer usage 仍然偏集中。

`v31_current_verify` 验证集 usage：

- level-0: `[24, 56]`
- level-1: `[0, 0, 0, 6, 74, 0]`

`v31_internal_compact_verify_e50` 验证集 usage：

- level-0: `[20, 60]`
- level-1: `[0, 0, 0, 3, 77, 0]`

因此，当前主线不能写成“side 分离和 code usage 都已经解决”。更准确的结论是：

- side routing 成功
- code utilization 仍然不够健康
- dataset leakage 仍待处理

---

## 5. `scripts/matrix_vis`：后验解释层的关键数据

## 5.1 Toy 结果

`toy_leaf_to_rectangle_axis_x`：

- `RMSE = 0.3371`
- `MAE = 0.2644`

`toy_leaf_to_rectangle_axis_y`：

- `RMSE = 0.0350`
- `MAE = 0.0293`

这说明在当前观测定义下：

- `y` 轴更接近单一开口运动，更容易重建
- `x` 轴更像受边界限制的内部重排，更难

## 5.2 matrix-free 与 OSQP 的一致性

toy x 轴：

- OSQP `RMSE = 0.3371022`
- matrix-free `RMSE = 0.3371164`

真实 full341 对比：

- `x rmse = 1.0712e-05`
- `y rmse = 2.1765e-05`
- `xy rmse = 2.4259e-05`

这说明：

- matrix-free 近似在当前问题上已经足够精确
- 可以用更低代价完成 subset / preview 级解释

## 5.3 真实数据运行代价

full341 + OSQP：

- `341` points
- `57970` observations
- 单轴运行时间约 `119s`

mouth119 + matrix-free：

- `119` points
- `7021` observations
- x 轴约 `4.31s`
- y 轴约 `11.77s`

因此当前最实用的使用方式是：

- full341 用于单例高保真分析
- mouth119 + matrix-free 用于快速解释与预览

---

## 6. 当前总判断

### 6.1 已经站住的部分

1. `lq` 已经给出足够清晰的结构探索链条
2. `disentangleNet/v31` 已经稳定实现 side-aware 路由
3. `matrix_vis` 已经是一个可信的后验解释工具，而不只是概念原型

### 6.2 仍然未解决的部分

1. free branch 的 code usage 仍然偏集中
2. free / private 中的 dataset leakage 明显存在
3. severity 信息还没有形成强、稳的可解释表示

### 6.3 当前最合理的论文式叙述

当前最稳的叙述不是“已经完成面瘫分级”，而是：

> 我们已经建立了一条 side-aware、可后验解释的共享运动分解主线。  
> 这条主线在 side 路由上表现强，但在 dataset-invariance 与 severity 表示上仍存在明显缺口。
