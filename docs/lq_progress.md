# LQ 原型进展笔记

最后更新：2026-04-20（第一轮 side semantic bank 冒烟测试）

## 范围

本文档记录 `scripts/lq` 最近的实现和实验进展，聚焦当前单方向设置：

- 模式：`x`
- 区域：`mouth`
- 数据：`data/win20-step20/IMR,data/win20-step20/TT`
- 输入：`B x T x 1 x H x W`
- 当前工作目标：共享离散运动 basis 学习 + 逐帧重建损失

它是 [`RESEARCH_PROGRESS.md`](/home/weizilin/generate_idea/RESEARCH_PROGRESS.md) 的配套工作笔记。

第一轮重构说明：

- `scripts/lq` 内部现已拆分为 `training/`、`data/` 和 `model/` 子包
- 公共入口点保持不变：
  - `python scripts/lq/train.py ...`
  - `python scripts/lq/analyze_checkpoint.py ...`
- 当前数据集样本字段、checkpoint 字段和运行脚本语义有意保持不变

重要说明：

- 以下记录的 `v1` 到 `v10` 指标产生于早期 `win10-step10` 轮次
- 标准数据集根目录现已切换到 `win20-step20`
- FSQ 基线已在新的数据集设置下重跑，应作为未来工作的权威比较点

当前标准基线：

- 运行：`outputs/lq_x_mouth_v10_fsq_probe_win20`
- `val_loss = 0.3619`
- `val_recon = 0.3600`
- `val_shared_recon = 0.3620`
- `val_scaled_residual = 0.0034`
- code usage：
  - L1 `[20, 60]`
  - L2 `[20, 23, 37]`
  - L3 `[18, 2, 3, 3, 25, 29]`

重要注意事项：

- `win20-step20` 验证划分当前仅包含 `80` 个有效帧
- 新基线仍显示广泛的 FSQ code usage，但该证据基于比旧 `win10-step10` 轮次小得多的验证集

`win20` 上第一个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v11_private_dim8_probe_win20`
- 变化：`private_dim 32 -> 8`
- 结果：
  - `val_loss = 0.3627`
  - `val_recon = 0.3610`
  - `val_shared_recon = 0.3625`
  - `val_scaled_residual = 0.0026`
  - L2 坍缩到 `[0, 80, 0]`
- 决策：
  - 拒绝此方向作为下一基线
  - 在当前 `win20 + FSQ` 设置下，直接缩小 private 潜在宽度破坏性太大

`win20` 上第二个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v12_private_decoder16_probe_win20`
- 变化：private decoder 隐藏宽度从基线有效值 `64` 降到 `16`
- 结果：
  - `val_loss = 0.3636`
  - `val_recon = 0.3615`
  - `val_shared_recon = 0.3620`
  - `val_scaled_residual = 0.0039`
  - L2 `[20, 22, 38]`
  - L3 `[17, 3, 2, 3, 17, 38]`
- 决策：
  - 也拒绝此方向
  - 它避免了 `v11` 中看到的严重 L2 坍缩，但仍未能超越 `v10` 基线

`win20` 上第一个 FSQ 时代辅助探测：

- 运行：`outputs/lq_x_mouth_v13_side_cont_probe_win20`
- 变化：启用 `side_cont_weight=0.15`，同时保持 `side_disc_weight=0.0`
- 结果：
  - `val_recon = 0.3623`
  - `val_shared_recon = 0.3631`
  - `val_scaled_residual = 0.0018`
  - `val_side_cont = 1.0522`
  - L2 `[46, 34, 0]`
  - L3 `[31, 28, 9, 11, 1, 0]`
- 决策：
  - 目前拒绝此方向
  - 连续 side 监督在当前 `win20 + FSQ` 基线上未能改善 shared-motion 重建

`win20` 上第三个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v14_level_qr_probe_win20`
- 变化：用严格的 level 内 QR 正交化替换逐 basis 归一化
- 结果：
  - `val_recon = 0.3574`
  - `val_shared_recon = 0.3604`
  - `val_scaled_residual = 0.0049`
  - L2 `[21, 59, 0]`
  - L3 `[19, 2, 2, 2, 5, 50]`
- 决策：
  - 混合结果，不直接推广
  - QR 改善了重建和 shared 重建，但 code usage 变得更集中

当前后续调整：

- `level_qr` 仅移除每个 level 内的相似性
- 在 `scripts/lq/model/network.py` 中添加了新的 `global_qr` 模式
- `global_qr` 对所有 11 个 basis 做一次 QR，因此跨 level 的 basis 相似性不再被允许
- 探测入口：`scripts/lq/run_train_x_mouth_v15_global_qr_probe.sh`

`win20` 上第四个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v15_global_qr_probe_win20`
- 变化：用对所有 11 个 basis 的一次全局 QR 替换 level -wise QR
- 结果：
  - `val_loss = 0.3588`
  - `val_recon = 0.3572`
  - `val_shared_recon = 0.3597`
  - `val_scaled_residual = 0.0041`
  - L1 `[18, 62]`
  - L2 `[19, 5, 56]`
  - L3 `[58, 22, 0, 0, 0, 0]`
- 决策：
  - 作为有用的结构确认处理，而非新基线
  - 完整 bank QR 进一步改善了总重建，但离散 usage 比 `v14` 更集中

`win20` 上第五个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v16_global_qr_basis_l1_probe_win20_e30`
- 变化：保持完整 bank QR 并在 QR 后对结构化 basis bank 添加 basis L1 稀疏惩罚
- 结果：
  - `val_loss = 0.3478`
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3580`
  - `val_scaled_residual = 0.0353`
  - `val_basis_l1 = 0.00266`
  - L1 `[56, 24]`
  - L2 `[18, 7, 55]`
  - L3 `[0, 0, 0, 0, 7, 73]`
- 决策：
  - 不作为可解释性基线推广
  - 稀疏性先验在数值上有效，但模型通过允许 private 残差分支大得多来补偿

`win20` 上第六个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v17_residual_fsq_basis_l1_probe_win20_e50`
- 变化：在保持 `global_qr + basis_l1` 的同时，用残差 stage 方式 FSQ 替换单个 FSQ
- 结果：
  - `val_loss = 0.3291`
  - `val_recon = 0.3109`
  - `val_shared_recon = 0.3423`
  - `val_scaled_residual = 0.0393`
  - `val_basis_l1 = 0.00239`
  - L1 `[22, 58]`
  - L2 `[50, 8, 22]`
  - L3 `[55, 2, 1, 3, 13, 6]`
- 决策：
  - 不作为可解释性基线推广
  - residual FSQ 比 `v16` 更广泛地分散了高层 code usage，但总改善仍伴随着 shared 重建变差和 private 残差分支过大

`win20` 上第七个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v18_residual_fsq_sparse_shared_probe_win20_e50`
- 变化：保持 residual FSQ，用 anchor 引导的稀疏共享混合增加 shared-path 容量，并添加对 `shared_recon` 的直接监督
- 结果：
  - `val_recon = 0.3150`
  - `val_shared_recon = 0.3469`
  - `val_scaled_residual = 0.0394`
  - `val_basis_l1 = 0.00251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 8, 12]`
- 决策：
  - 有前途的正向方向，但还不是可解释性基线
  - 与 `v17` 相比，shared 重建明显改善，但 private 残差仍然过大

`win20` 上第八个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe_win20_e50`
- 变化：保持 `v18` shared 设计并将 `private_residual_max_l1` 从 `1.0` 收紧到 `0.5`
- 结果：
  - `val_recon = 0.3275`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0214`
  - `val_basis_l1 = 0.00252`
  - L1 `[22, 58]`
  - L2 `[58, 0, 22]`
  - L3 `[57, 1, 1, 1, 10, 10]`
- 决策：
  - 有前途的可解释性权衡
  - 与 `v18` 相比，private 残差大幅下降，而 shared 重建仍比 `v17` 好得多

`win20` 上第九个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v20_residual_fsq_sparse_shared_privatecap04_probe_win20_e50`
- 变化：保持 `v19` 结构并将 `private_residual_max_l1` 从 `0.5` 进一步收紧到 `0.4`
- 结果：
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3448`
  - `val_scaled_residual = 0.0171`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 17, 3]`
- 决策：
  - 有用的更严格 private 变体，但不是对 `v19` 的干净替换
  - 它将 `scaled_residual` 压得比 `v19` 低，但 `shared_recon` 略差且 L3 usage 更集中

`win20` 上第十个 FSQ 时代结构探测：

- 运行：`outputs/lq_x_mouth_v21_residual_fsq_sparse_shared_privatecap06_probe_win20_e50`
- 变化：保持 `v19` 结构并将 `private_residual_max_l1` 从 `0.5` 放宽到 `0.6`
- 结果：
  - `val_recon = 0.3245`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 11, 9]`
- 决策：
  - 拒绝作为下一可解释性基线
  - 它改善了普通重建，但主要通过允许 private 分支再次增长；shared 重建并未比 `v19/v20` 有意义地更好

`v19` 附近当前局部 sweep 读数：

- `cap=0.4` 将 `scaled_residual` 压得最低，但更高层 usage 更集中
- `cap=0.5` 仍是最安全的选择，因为它在 private 抑制和 L3 分散之间保持了更好平衡
- `cap=0.6` 对当前目标太宽松，主要恢复了 private 修正能力

`win20` 上第一轮 side semantic bank 冒烟测试：

- 运行：`outputs/lq_x_mouth_v22_side_semantic_bank_probe_smoke`
- preset 变化：
  - 保持 `v19` 主干不变：
    - `action_basis_init_path=scripts/lq/init_basis/basis_x.npy`
    - `shared_recon_weight=1.0`
    - `quantizer_type=residual_fsq`
    - `basis_orthogonalization=global_qr`
    - `private_residual_max_l1=0.5`
  - 仅启用第一轮 side semantic bank：
    - `side_semantic_enabled=True`
    - `side_basis_count=2`
    - `side_loss_weight=0.3`
- 冒烟测试结果：
  - `val_loss = 1.0672`
  - `val_recon = 0.3636`
  - `val_shared_recon = 0.3636`
  - `val_scaled_residual = 0.00233`
  - `val_side_group = 1.1104`
- 分析结果：
  - 写入的输出文件：
    - `analysis/summary.json`
    - `analysis/side_basis_bank_heatmap.png`
    - `analysis/group_level_representations.npz`
  - `side_basis_shape = [2, 119, 119]`
  - `mean_side_path_usage = 0.500`
  - `mean_free_path_usage = 0.273`
  - `mean_side_recon_l1 = 0.00162`
  - `mean_free_recon_l1 = 0.000262`
  - `side_from_side_rep_acc = 0.364`
  - `side_from_free_rep_acc = 0.364`
  - `dataset_from_side_rep_acc = 0.818`
  - `dataset_from_free_rep_acc = 0.818`
- 决策：
  - 接受为成功的第一轮 preset 和分析冒烟测试
  - 尚不能解读为语义分离成功
  - `B_side` 并非空闲，但经过一个 epoch 后它未能在 side probe 上超越 free path，且两个 shared 分支仍保留明显的 dataset 信号

## 当前实现状态

### 1. 数据集/输入流水线

当前状态：

- `scripts/lq/datasets.py` 现在支持用于训练的分组序列加载。
- 训练输入对齐到 `batch x win x matrix_size x matrix_size`。
- 当前实验使用 `group_size=4`。
- 训练开始前已添加 batch 内存冒烟测试。

已验证：

- 搭配 `batch_size=64`、`group_size=4`、`mode=x`、`region=mouth` 时，一个输入 batch 形状为 `(64, 4, 1, 119, 119)`。
- 输入张量本身约为 `13.83 MiB`。
- 前向+反向冒烟测试通过，无 OOM。

已知剩余数据集问题：

- `deleted_x` / `deleted_y` 语义尚未完全集成。
- 数据集元数据对下游分析仍不足。
- 采样尚未按患者或数据集平衡。

相关检查清单：

- [`docs/lq_dataset_refactor_checklist.md`](/home/weizilin/generate_idea/docs/lq_dataset_refactor_checklist.md)

### 2. 训练流水线

已实现：

- `scripts/lq/train.py` 现在直接接受序列输入。
- 损失按帧计算，然后以 valid/padding mask 归约。
- 当前训练阶段默认要求 `basis_init`。
- 完整训练前运行 batch 内存验证。
- 第一轮重构将训练内部拆分到 `scripts/lq/training/`，同时保留当前 CLI 签名和指标 key

现在跟踪的指标包括：

- `loss`
- `recon`
- `shared_recon`
- `lq`
- `orth`
- `residual`
- `scaled_residual`
- 可选的 side / dataset loss

### 3. 模型结构

当前模型文件：

- [`scripts/lq/model/network.py`](/home/weizilin/generate_idea/scripts/lq/model/network.py)
- `network.py` 现在是对拆分模型模块 `scripts/lq/model/` 的薄兼容层

本轮实现结构变化：

- 序列输入展平/恢复逻辑
- 可配置 `pool_size`
- 可配置 `shared_dim`
- 带 `private_residual_max_l1` 的有上限 private 残差
- 可选带 anchor bias 的软 basis 混合
- 官方量化器开关：
  - `quantizer_type="latent_quantize"`
  - `quantizer_type="fsq"`

当前工作解读：

- shared path：离散运动码 + action basis 重建
- private path：残差修正分支

### 4. 分析/工具

已实现：

- `scripts/lq/analyze_checkpoint.py` 现在可以加载带有 `pool_size`、`shared_dim`、量化器配置和残差上限配置的更新 checkpoint。
- basis bank 热图和 code-usage 汇总自动生成。
- Pillow 重采样兼容性警告已移除。
- 第一轮重构保持分析 CLI 不变，同时将导入切换到拆分的 `data/` 和 `model/` 包

## 实验时间线

除非另有说明，以下所有实验使用相同的基础设置：

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`

### v1. 保守基线

输出：

- `outputs/lq_x_mouth_v1`

结果：

- `val_loss = 0.6309`
- code usage 立即出现坍缩：
  - L1 `[0, 913]`
  - L2 `[0, 913, 0]`
  - L3 `[0, 0, 0, 0, 788, 125]`

结论：

- 基线能训练，但坍缩立即出现。

### v2. 官方 LatentQuantize 抗坍缩探测

变化：

- 切换到官方的 `LatentQuantize`
- 使用抗坍缩导向设置，如更强的权重衰减和 `optimize_values=False`

输出：

- `outputs/lq_x_mouth_v2_official_lq_anticollapse`

结果：

- `val_loss = 0.6308`
- code usage 仍然坍缩：
  - L1 `[0, 913]`
  - L2 `[0, 913, 0]`
  - L3 `[0, 0, 0, 0, 803, 110]`

结论：

- 官方 LQ 设置单独未能解决坍缩。

### v3. 无离散 side + 更强 LQ

变化：

- 移除离散 side 监督
- 增加 LQ 压力

输出：

- `outputs/lq_x_mouth_v3_no_disc_side_stronger_lq`

结果：

- `val_loss = 0.4414`
- L3 usage 仍几乎单码：`[0, 0, 0, 0, 903, 10]`

结论：

- 更好的验证损失，但坍缩仍然严重。

### v4. 低残差探测

变化：

- 降低 private 残差贡献权重

输出：

- `outputs/lq_x_mouth_v4_low_residual`

结果：

- `val_loss = 0.4974`
- L3 `[0, 0, 0, 0, 898, 15]`

结论：

- 降低残差权重未能修复坍缩。
- 模型仍能利用残差幅度。

### v5. 低残差有上限探测

变化：

- 用 `private_residual_max_l1=1.0` 限制 private 残差均值绝对幅度

输出：

- `outputs/lq_x_mouth_v5_low_residual_capped`

结果：

- `val_loss = 0.5112`
- L3 `[0, 0, 0, 0, 896, 17]`

结论：

- 残差旁路被约束，但坍缩仍然存在。

### v6. 无侧别探测

变化：

- 完全禁用 side 监督

输出：

- `outputs/lq_x_mouth_v6_no_side_probe`

结果：

- `val_loss = 0.3462`
- `val_recon = 0.3255`
- `val_shared_recon = 0.3541`
- `val_scaled_residual = 0.0392`
- code usage：
  - L1 `[0, 913]`
  - L2 `[0, 913, 0]`
  - L3 `[0, 0, 0, 0, 891, 22]`

结论：

- side 监督不是坍缩的主要原因。
- 这成为主要结构比较基线。

### v7. 共享瓶颈探测

变化：

- 将 `shared_dim` 从 `32` 缩小到 `8`

输出：

- `outputs/lq_x_mouth_v7_shared_bottleneck`

结果：

- `val_loss = 0.3446`
- L3 完全坍缩到一个码：`[0, 0, 0, 0, 0, 913]`

结论：

- 单独收窄 shared 潜在使坍缩更严重。

### v8. Pool-2 探测

变化：

- 将 `AdaptiveAvgPool2d((1, 1))` 替换为 `AdaptiveAvgPool2d((2, 2))`

输出：

- `outputs/lq_x_mouth_v8_pool2_probe`

结果：

- `val_loss = 0.3588`
- L2 略分散：`[17, 896, 0]`
- L3 仍坍缩：`[0, 0, 0, 0, 913, 0]`

结论：

- 池化尺度影响早期行为，但未能从根本上解决坍缩。

### v9. 软 Basis 探测

变化：

- 用软混合替换硬逐 level basis 选择，使用离散选中码作为 anchor bias

输出：

- `outputs/lq_x_mouth_v9_soft_basis_probe`

结果：

- `val_loss = 0.3458`
- L3 `[0, 0, 0, 0, 904, 9]`

结论：

- 仅放宽 basis 选择是不够的。

### v10. 官方 FSQ 替换

变化：

- 用官方 `FSQ` 替换 `LatentQuantize`
- 保持其余 v6 风格结构约束可比较

输出：

- `outputs/lq_x_mouth_v10_fsq_probe`

结果：

- `val_loss = 0.3238`
- `val_recon = 0.3081`
- `val_shared_recon = 0.3335`
- `val_scaled_residual = 0.0329`
- code usage：
  - L1 `[361, 552]`
  - L2 `[335, 78, 500]`
  - L3 `[300, 33, 22, 36, 38, 484]`

结论：

- 这是第一个不仅改变重建指标，而且明确改善 code usage 的实验。
- FSQ 不仅稳定运行，而且在当前架构下显著减少坍缩。
- 在旧的 `win10-step10` 轮次中，`v10` 成为新基线。
- 在当前 `win20-step20` 轮次中，重跑仍显示广泛的 code usage 并保持为活跃基线，但其验证集小得多。

## 当前主要发现

### 未能修复坍缩的因素

- side 监督变化
- 更强的官方 `LatentQuantize` 设置
- 仅降低 private 残差权重
- 仅残差上限
- 缩小 shared 潜在维度
- 仅增加池化大小
- 仅软 basis 混合

### 起作用的因素

- 将 shared 量化器切换到官方 `FSQ`

在当前阶段，证据支持此解读：

- 早期的坍缩不仅仅是 loss 权重问题
- 它也不是由 `1x1` 池化或 side 监督等单一结构细节所能解释的
- 量化器的选择是 shared 离散路径是否能有意义地使用其码本的主要因素

## 当前风险和开放问题

1. `v10` 大大改善了 code usage，但 shared path 仍弱于完整重建路径。
   - `val_shared_recon = 0.3335`
   - `val_recon = 0.3081`

2. 改善仍与非平凡 private 残差分支共存。
   - 结构比之前更健康，但解耦尚未得到证明。

3. FSQ 当前在我们的指标布局中没有显式 LQ 惩罚项。
   - 在当前实现中，`lq_loss_per_sample` 在 FSQ 路径上为零。
   - 这对当前比较是可接受的，但应在后续分析中保持显式。

4. 数据集语义尚未完全确定。
   - 尤其是 `deleted_x` / `deleted_y`

## 建议的下一步

将 `v10 FSQ` 在 `win20-step20` 上作为活跃基线，然后在其上继续结构分析，而非回到 `LatentQuantize`。

下一个合理的问题是：

1. shared path 是否应相对于 private 残差 path 进一步增强
2. side / dataset 辅助头是否应在 FSQ 之上重新引入
3. 相同趋势是否对 `mode=y` 和后续 `full` 区域设置成立
