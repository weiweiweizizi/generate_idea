# Idea Experiments Log

[TOC]

**Last Updated**: 2026-04-29

## 0. 文档定位

这份文档是当前研究最详细的实验账本，只保留与当前主线直接相关的内容：

- `scripts/lq`
  - 作为 `disentangleNet` 之前的结构探索与消融试验场
- `scripts/disentangleNet`
  - 作为当前接受的、冻结的 `v31` 训练与分析栈
- `scripts/matrix_vis`
  - 作为把单轴距离差观测解释成局部运动轨迹的后验可视化与诊断工具

本次更新删除了大量已经明显过时的内容，尤其是：

- 早期“推荐想法排名”
- 已被后续结果覆盖的旧版 next-step 列表
- 把 `scripts/lq` 与 `scripts/disentangleNet` 混写为同一阶段状态的表述

---

## 1. 当前主线结构

### 1.1 三条线各自负责什么

1. `scripts/lq`
   - 负责回答“结构上怎么改，shared / side / private 三条路径会发生什么”
   - 这里保留了从 collapse、FSQ 替换、shared 结构增强、side branch 引入到 severity probe 的完整试错链条

2. `scripts/disentangleNet`
   - 负责把已经接受的 `v31` 结构冻结下来
   - 当前只保留 `v31` 所需的训练入口、模型、数据、初始化基和 post-hoc probe 分析

3. `scripts/matrix_vis`
   - 负责把一个单轴 `diff of distance matrix` 解释成一个窗口内的局部运动轨迹
   - 它不是训练模型，而是对 basis / observation 的后验解释工具

### 1.2 当前对整体路线的理解

- `lq` 已经完成“结构探索”阶段
- `disentangleNet` 是当前“可复核主线”
- `matrix_vis` 是当前“解释层工具”

因此，当前研究不是“继续发散找新分解法”，而是围绕下面三个问题收束：

1. side 信息能否被稳定路由到单独的 side branch
2. free / private 中还残留多少 dataset 痕迹
3. basis 或窗口差观测如何被解释成可视的局部运动

---

## 2. `scripts/lq`：前置探索时间线

当前 `lq_*` 运行目录已统一整理到：

- `outputs/lq/win20-step20/`

当前仓库实际保留的目录索引如下：

| 阶段 | 目录 |
|------|------|
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

说明：

- `v10-v19` 仍然是重要的历史结论，但当前仓库里主要保留为文档记录
- 当前磁盘上的 `lq` 运行目录索引从 `v20` 开始

## 2.1 Collapse 阶段：`v1` 到 `v9`

这一阶段的主要结论很明确：

- 仅靠早期 LQ 设置无法避免 collapse
- 调 side loss、调 residual weight、减小 shared 容量，都不能根治 collapse
- shared 离散码没有被稳定使用，模型会持续走 private residual 这条逃逸路径

代表性结果：

| 版本 | 核心改动 | 关键结果 | 结论 |
|------|----------|----------|------|
| `v1` | 保守 baseline | `val_loss=0.6309`，L3 几乎单码使用 | 立即 collapse |
| `v2` | 官方 `LatentQuantize` anti-collapse 设置 | `val_loss=0.6308`，usage 仍塌缩 | 单换 quantizer 不够 |
| `v3` | 去掉 discrete side，增强 LQ 压力 | `val_loss=0.4414`，L3 仍近单码 | loss 改善不代表 shared code 健康 |
| `v4` | 降低 private residual weight | `val_loss=0.4974` | 只减 residual 权重没用 |
| `v5` | 加 `private_residual_max_l1=1.0` | `val_loss=0.5112` | 限 residual 幅度后仍塌缩 |
| `v6` | 彻底关掉 side supervision | `val_recon=0.3255`，L3 仍高度集中 | collapse 不是 side loss 单独造成的 |
| `v7` | 缩 shared bottleneck | 保持较差 usage | 直接压 shared 容量是错误方向 |
| `v8` | pool size 改为 `2x2` | 无根本改善 | 仅保留粗空间布局不够 |
| `v9` | soft basis probe | 仍非稳定解 | 说明问题不在 basis 混合方式本身 |

### 2.1.1 阶段性判断

- 模型的首要问题不是“side 监督太强”
- 也不是“shared 维度还不够小”
- 真正的问题是 shared 离散路径缺少稳定被使用的结构条件

---

## 2.2 FSQ 切换与 shared 结构增强：`v10` 到 `v21`

`v10` 是一个关键分界点。它不是最终结构，但它是第一次让 code usage 明显好转的版本。

### 2.2.1 `v10`：第一个可接受的 anti-collapse baseline

关键指标：

- `val_loss = 0.3619`
- `val_recon = 0.3600`
- `val_shared_recon = 0.3620`
- `val_scaled_residual = 0.0034`
- code usage
  - L2: `[20, 23, 37]`
  - L3: `[18, 2, 3, 3, 25, 29]`

结论：

- 官方 `FSQ` 替换是第一步真正缓解 collapse 的因素
- 从这里开始，后续问题从“码完全不用”转为“码虽然用了，但解释性是否足够”

### 2.2.2 `v11` 到 `v16`：直接压 private 或强化 basis 约束都不够

| 版本 | 核心改动 | 关键指标 | 结论 |
|------|----------|----------|------|
| `v11` | `private_dim 32 -> 8` | `val_recon=0.3610`，L2 变成 `[0,80,0]` | 直接压 private 太粗暴，反而更坏 |
| `v12` | 缩 private decoder hidden width | `val_recon=0.3615` | 没有优于 `v10` |
| `v13` | side continuous probe | `val_recon=0.3623` | side continuous 不改善 shared 解释 |
| `v14` | `level_qr` | `val_recon=0.3574`，shared 稍好，usage 更集中 | QR 有益但副作用明显 |
| `v15` | `global_qr` | `val_recon=0.3572`，L3 更集中 | 全局 QR 提升重建但牺牲 usage spread |
| `v16` | `global_qr + basis_l1` | `val_recon=0.3310`，`val_scaled_residual=0.0353` | 总重建下降主要靠 private residual 变大 |

这一段最重要的经验不是某个版本“赢了”，而是：

- 更强的 basis 正交或稀疏约束会改善重建
- 但这些收益很容易被 private residual 吞掉
- 因此后面必须同时看 `shared_recon` 和 `scaled_residual`

### 2.2.3 `v17` 到 `v21`：Residual FSQ + shared 监督开始形成主线

| 版本 | 核心改动 | `val_recon` | `val_shared_recon` | `val_scaled_residual` | 结论 |
|------|----------|-------------|--------------------|-----------------------|------|
| `v17` | `residual_fsq + global_qr + basis_l1` | `0.3109` | `0.3423` | `0.0393` | higher-level usage 变好，但 private residual 过大 |
| `v18` | `v17 + sparse_shared_mixing + shared_recon_loss` | `0.3150` | `0.3469` | `0.0394` | shared reconstruction 明显回升 |
| `v19` | `v18 + tighter private cap=0.5` | `0.3275` | `0.3446` | `0.0214` | 当前这一阶段最好的 interpretability tradeoff |
| `v20` | `v19 + cap=0.4` | `0.3310` | `0.3448` | `0.0171` | private 更低，但 usage 更集中 |
| `v21` | `v19 + cap=0.6` | `0.3245` | `0.3446` | `0.0251` | private 又变大，不如 `v19` 稳 |

阶段性结论：

1. 只压 private branch 没用
2. residual FSQ 本身是有效结构改动
3. “shared 结构增强 + 直接优化 `shared_recon`”是正确方向
4. `v19` 是从纯结构探索走向 side-aware 分支之前最稳的解释性 baseline

---

## 2.3 Side semantic bank：`v22` 与 `v23`

这一步开始显式问一个新问题：

> 能不能把 side 语义从 shared/free 里剥出来，交给单独的 side basis bank？

### 2.3.1 `v22`：round-1 side semantic bank

运行目录：

- `outputs/lq/win20-step20/lq_x_mouth_v22_side_semantic_bank_probe_win20_e50`

结构变化：

- 保留 `v19` backbone
- 打开 `side_semantic_enabled=True`
- `side_basis_count=2`
- `side_loss_weight=0.3`

验证指标：

- `val_recon = 0.3330`
- `val_shared_recon = 0.3504`
- `val_scaled_residual = 0.0216`
- `mean_side_path_usage = 0.500`
- `mean_free_path_usage = 0.2727`
- `side_from_side_rep_acc = 0.4545`
- `side_from_free_rep_acc = 0.4545`
- `dataset_from_side_rep_acc = 0.8182`
- `dataset_from_free_rep_acc = 0.8182`

结论：

- side path 已经被用上了，不是 idle branch
- 但 2 个 side basis 还不能形成有效的 side separation
- free / side 两条 shared 路径都还带有明显 dataset 痕迹

### 2.3.2 `v23`：side subspace orth 失败

运行目录：

- `outputs/lq/win20-step20/lq_x_mouth_v23_side_subspace_orth_probe_win20_e50`

关键结果：

- `val_recon = 0.3342`
- `val_shared_recon = 0.3504`
- `val_scaled_residual = 0.0208`
- `side_from_side_rep_acc = 0.5227`
- `side_from_free_rep_acc = 0.5182`
- `dataset_from_side_rep_acc = 0.7818`
- `dataset_from_free_rep_acc = 0.8182`
- `raw_linear_r2_free_to_side ≈ 1.0`
- `raw_linear_r2_side_to_free ≈ 1.0`

结论：

- `subspace_orth` 没有真正把 side / free 拉开
- 两个 latent 子空间几乎线性等价
- 因此这条路线被放弃

---

## 2.4 Laterality 强化与早分支化：`v29` 到 `v31`

这一段的主题是：

> side branch 不是“有无”的问题，而是要不要更强的 laterality 结构、要不要更早切分。

### 2.4.1 `v29`：laterality contrast probe

运行目录：

- `outputs/lq/win20-step20/lq_x_mouth_v29_laterality_contrast_probe_win20_e50`

关键现象：

- `side_basis_count = 4`
- `side_from_usage_acc = 0.6773`
- `side_from_coeff_acc = 0.5955`
- `side_from_usage_coeff_acc = 0.7227`
- `dataset_from_usage_acc = 0.8182`

按 side 聚合的 usage 平均值出现明显 laterality 分布：

- Left:
  - `usage_b0 = 0.660`
- Right:
  - `usage_b1 = 0.774`
- Normal:
  - 在 `b0/b2/b3` 之间更分散

结论：

- laterality contrast 开始出现“左侧基 / 右侧基”的结构趋势
- 但 dataset 偏差仍然明显
- 4 个 side basis 的设计偏重 laterality，但还不够稳

### 2.4.2 `v30`：收敛到 3 个 side basis

运行目录：

- `outputs/lq/win20-step20/lq_x_mouth_v30_joint_qr_levels26_side3_probe_win20_e50`

关键结果：

- `side_basis_count = 3`
- `side_from_usage_acc = 0.8318`
- `side_from_coeff_acc = 0.7682`
- `side_from_usage_coeff_acc = 0.8045`
- `dataset_from_usage_acc = 0.8182`

按 side 聚合的 usage 平均值：

- Left:
  - `usage_b2 = 0.859`
- Normal:
  - `usage_b1 = 0.367`
- Right:
  - `usage_b0 = 0.310`, `usage_b1 = 0.292`, `usage_b2 = 0.398`

结论：

- 3-basis 结构比 `v29` 更简洁，也更接近后续固定版
- 但这仍然是 `scripts/lq` 阶段的版本，不是当前冻结主线

### 2.4.3 `v31`：旧 `lq` 版与冻结 `disentangleNet` 版要分开看

这是本次文档更新中最需要澄清的地方。

`scripts/lq` 里仍保留旧的：

- `outputs/lq/win20-step20/lq_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50`

它延续的是 `lq` 阶段的实验轨迹，分析文件里还能看到旧的 11-basis 结构痕迹。

而 `scripts/disentangleNet` 冻结下来的 `v31` 已经是新的接受版本：

- `levels = 2,6`
- `side_basis_count = 3`
- `early_branch_factorization = True`
- `basis_orthogonalization = joint_global_qr`
- `side_pooling = fixed_region2_contrast`

也就是说：

- `lq/v31` 是探索期版本
- `disentangleNet/v31` 是冻结后的接受版本

这一点在后面的 `scripts/disentangleNet` 小节中单独展开。

---

## 2.5 Severity 辅助线：`v32`

运行目录：

- `outputs/lq/win20-step20/lq_x_mouth_v32_joint_qr_levels26_side3_sparse_side_severity_probe_win20_e50`

这个版本主要回答：

> side-aware level-2 representation 是否已经足够承载 severity 信息？

### 2.5.1 结果

当前实际形成的是二分类子问题：

- `severity_counts`
  - `Normal = 111`
  - `Mild = 182`

关键 probe：

| 任务 | Accuracy | Balanced Acc | Macro F1 |
|------|----------|--------------|----------|
| `severity_from_level2_coeff` | `0.6212` | `0.5000` | `0.3832` |
| `severity_from_level2_rep` | `0.6621` | `0.5699` | `0.5377` |
| `severity_from_level2_usage` | `0.6621` | `0.5699` | `0.5377` |
| `severity_from_level2_usage_coeff` | `0.6621` | `0.5699` | `0.5377` |
| `severity_from_free_rep` | `0.6519` | `0.5616` | `0.5304` |
| `side_from_level2_rep` | `0.4198` | `0.3807` | `0.3063` |

### 2.5.2 结论

- severity 信息在当前 setup 下是弱信号
- `level2` side-aware 表示对 severity 的提升非常有限
- `severity_from_level2_coeff` 甚至退化到 balanced accuracy `0.5`
- 因此当前不能把 `v32` 写成“severity 已被有效解耦”的证据

更准确的表述应当是：

> `v32` 只说明 side-aware 中层表示对 severity 可能有弱相关，但离可用结论还很远。

---

## 3. `scripts/disentangleNet`：当前接受的冻结 `v31`

## 3.1 包结构定位

`scripts/disentangleNet` 不是一个继续发散试验的目录，而是：

- 从 `scripts/lq` 中抽出的、当前接受的 `v31` 训练栈
- 只保留 `v31` 所需配置面
- 保留对应的后处理分析路径

入口：

- 训练：
  - `bash scripts/disentangleNet/run_train_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe.sh`
- 分析：
  - `analysis/analyze_checkpoint.py`
  - `analysis/analyze_side_interpretability.py`
  - `analysis/analyze_kfold_report.py`

## 3.2 冻结配置

当前 `train.py` 中写死的 `V31_FIXED_CONFIG` 关键设置如下：

| 项目 | 值 |
|------|----|
| mode | `x` |
| region | `mouth` |
| data | `data/win20-step20/IMR,data/win20-step20/TT` |
| group_size | `4` |
| levels | `2,6` |
| quantizer | `residual_fsq` |
| basis orth | `joint_global_qr` |
| shared soft mixing | `True` |
| shared topk | `2` |
| side branch | `enabled` |
| side basis count | `3` |
| side pooling | `fixed_region2_contrast` |
| side loss | group-level only (`side_loss_weight=0.3`) |
| private residual cap | `0.5` |
| early branch factorization | `True` |

### 3.2.1 与 `lq` 的区别

冻结版 `disentangleNet/v31` 与旧 `lq/v31` 的关键差异：

- 去掉了继续开放的大量实验开关
- 固定到 `levels=2,6`
- 固定为 early-branch factorization
- 固定为 3 个 side basis

因此后续如果写“当前主线”，应默认指向 `scripts/disentangleNet`，而不是 `scripts/lq`。

## 3.3 当前验证产物

当前有三组相关输出：

- 旧导出：
  - `outputs/disentangleNet/v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50`
- 核验 rerun：
  - `outputs/disentangleNet/v31_current_verify`
- 紧凑核验 rerun：
  - `outputs/disentangleNet/v31_internal_compact_verify_e50`

这三组输出的定性结论一致，因此这里重点记录“稳定复现出来的模式”，而不是把某一组当成唯一真值。

## 3.4 训练与验证指标

`v31_current_verify`：

- `val_recon = 0.3097`
- `val_shared_recon = 0.3255`
- `val_scaled_residual = 0.0209`

`v31_internal_compact_verify_e50`：

- `val_recon = 0.3226`
- `val_shared_recon = 0.3377`
- `val_scaled_residual = 0.0199`

这两组数说明：

- 当前结构已经能把 plain reconstruction 压到 `0.31-0.32`
- shared reconstruction 在 `0.326-0.338` 区间
- private residual 被稳定压在 `~0.02`

## 3.5 Full post-hoc k-fold 结果

full k-fold 报告基于：

- `293` groups
- `267` subjects
- `5` folds
- stratification = `joint_side_dataset`

关键 probe 结果如下：

| 任务 | 结果范围 | 解读 |
|------|----------|------|
| `side_from_side_rep` | `0.9283 - 0.9317` acc | side branch 强烈编码了侧别 |
| `side_from_usage_coeff` | `0.9283 - 0.9420` acc | side 使用模式 + 系数几乎可直接读出侧别 |
| `side_from_free_rep` | `0.4710` acc | free branch 对 side 的显式判别能力明显更弱 |
| `dataset_from_side_rep` | `0.7713 - 0.7782` acc，balanced acc `0.50 - 0.53` | side branch 仍带 dataset 痕迹，但不稳定且偏类分布 |
| `dataset_from_free_rep` | `0.7986 - 0.8055` acc，balanced acc `0.645 - 0.694` | free branch 中 dataset 痕迹更明确 |
| `dataset_from_private_rep` | `0.8840 - 0.8908` acc，balanced acc `0.810 - 0.825` | private branch 是最强 dataset carrier |

### 3.5.1 当前最重要的结构性结论

1. side branch 已经成功承担了侧别语义
2. free branch 没有学成一个“纯 dataset-invariant shared motion branch”
3. private branch 仍然最强地承载 dataset 偏差

也就是说，当前 `v31` 不是“彻底解耦完成”，而是：

> 已经实现了 side-routing 成功，但 dataset leakage 仍未清理干净。

## 3.6 当前 code usage 的重要警告

虽然 `v31` 在 side probe 上表现很强，但 free quantizer 的 usage 仍然不健康。

`v31_current_verify` 的验证集 usage：

- level-0 counts: `[24, 56]`
- level-1 counts: `[0, 0, 0, 6, 74, 0]`

`v31_internal_compact_verify_e50` 的验证集 usage：

- level-0 counts: `[20, 60]`
- level-1 counts: `[0, 0, 0, 3, 77, 0]`

结论：

- 当前 `v31` 不是“code utilization 也已完美健康”
- 它更像是“side branch routing 已经明显成功，但 free codebook 仍偏集中”
- 因此后续若继续做主线改进，首要问题会从“能否分 side”转向“能否提升 free branch 的利用效率并降低 dataset leakage”

---

## 4. `scripts/matrix_vis`：后验轨迹解释工具

## 4.1 当前定位

`matrix_vis` 当前不是新的训练模型，而是：

- 给定单轴 `diff of distance matrix`
- 在 anchor、平滑和单轴观测假设下
- 恢复一个可解释的窗口内轨迹

它的价值不在于“证明轨迹唯一正确”，而在于：

1. 让 basis / observation 的语义可视化
2. 让我们知道哪些轴更容易、哪些区域更稳定
3. 提供真实数据上的后验 sanity check

## 4.2 Toy 实验：`y` 明显比 `x` 容易

### 4.2.1 `toy_leaf_to_rectangle_axis_x`

- `ground_truth_rmse = 0.3371`
- `ground_truth_mae = 0.2644`
- `ground_truth_max_abs_error = 0.9914`

### 4.2.2 `toy_leaf_to_rectangle_axis_y`

- `ground_truth_rmse = 0.0350`
- `ground_truth_mae = 0.0293`
- `ground_truth_max_abs_error = 0.0660`

结论：

- 在当前观测定义和约束下，`y` 轴几乎是一个简单开口问题
- `x` 轴更像固定边界下的内部重排问题
- 这与前面对面部运动的直觉一致，也与 README 中“`x` 通常比 `y` 难”的判断一致

## 4.3 Matrix-free solver 已经与 OSQP 对齐

### 4.3.1 Toy x 轴对比

`toy_leaf_to_rectangle_axis_x` 与 `toy_leaf_to_rectangle_axis_x_matrixfree` 的误差几乎相同：

- OSQP:
  - `rmse = 0.3371022`
- matrix-free:
  - `rmse = 0.3371164`

差异已经小到可以忽略。

### 4.3.2 真实 full341 对比

`outputs/matrix_vis/real_compare/imr_00228_win005_minus_win004_full341_osqp_vs_matrixfree/summary.json`

比较结果：

- `x rmse = 1.0712e-05`
- `y rmse = 2.1765e-05`
- `xy rmse = 2.4259e-05`
- `xy max_abs = 2.8926e-04`

结论：

- 在真实 full341 case 上，matrix-free 与 OSQP 的最终轨迹几乎重合
- 后续做 subset / preview / batch sanity check 时，matrix-free 是可接受近似

## 4.4 真实数据运行代价

### 4.4.1 Full341 + OSQP

以 `imr_00228_win005_minus_win004` 为例：

- 每轴 `341` points
- `57970` pairwise observations
- `20` time steps

运行时间：

- x 轴：
  - `run_time ≈ 119.80s`
- y 轴：
  - `run_time ≈ 119.22s`

### 4.4.2 Mouth119 + matrix-free

相同 case 的 mouth region：

- `119` points
- `7021` pairwise observations

运行时间：

- x 轴：
  - `run_time ≈ 4.31s`
- y 轴：
  - `run_time ≈ 11.77s`

结论：

- full341 级别更适合高保真单例分析
- mouth119 + matrix-free 已足够支持快速后验解释与可视预览

## 4.5 当前实际产物

当前 `matrix_vis` 已经能稳定导出：

- toy 单轴重建图
- toy 与 ground truth 对比图
- real 单轴轨迹图
- `x/y` 合成的 2D 预览
- `preview.gif`
- `snapshot_last_frame.png`
- `summary.json`
- `solution.npz`

例如：

- `outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_x_mouth_regions_anchor14/`
- `outputs/matrix_vis/real/imr_00228_win005_minus_win004_axis_y_mouth_regions_anchor14/`
- `outputs/matrix_vis/real/imr_00228_win005_minus_win004_compose_xy_mouth_regions/`

---

## 5. 当前主结论

1. `scripts/lq` 已经完成“从 collapse 到 side-aware branch”的探索阶段
2. `v19` 是前 side-semantic 阶段最稳的 interpretability baseline
3. `v22-v23` 说明“仅加 side bank / subspace orth”还不够
4. `v29-v31` 说明 laterality 结构确实可以被 side branch 显式承载
5. `scripts/disentangleNet` 冻结版 `v31` 已经稳定实现：
   - 强 side prediction
   - 较低 private residual
   - 但仍存在 free/private 的 dataset leakage
6. `v32` 说明 severity 仍不是当前结构的强项
7. `scripts/matrix_vis` 已经从概念工具变成可复核的后验解释工具
8. 当前下一阶段最值得继续追的问题不是“还能不能分出 side”，而是：
   - 能否减少 dataset leakage
   - 能否提升 free code usage 健康度
   - 能否让 `matrix_vis` 与 learned basis 的解释链条更紧密
