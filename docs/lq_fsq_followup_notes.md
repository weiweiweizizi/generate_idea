# LQ FSQ 后续跟进笔记

最后更新：2026-04-19

## 基线

- 标准数据集根目录：`data/win20-step20/IMR,data/win20-step20/TT`
- 当前目标基线运行：`outputs/lq_x_mouth_v10_fsq_probe_win20`
- 量化器：`FSQ`
- 模式：`x`
- 区域：`mouth`

## 历史背景

- 早期的 `v1` 到 `v10` 运行产生于旧的 `data/win10-step10/IMR,data/win10-step10/TT` 设置
- 这些运行仍可用于结构对比历史参考
- 它们不应被视为新的 `win20-step20` 数据集设置下的最终基线

## 探测日志

### 基线重跑：`win20-step20` 上的 `v10 FSQ`

- 状态：已完成
- 命令：

```bash
bash scripts/lq/run_train_x_mouth_v10_fsq_probe.sh
```

- 预期输出：
  - `outputs/lq_x_mouth_v10_fsq_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v10_fsq_probe_win20/analysis/summary.json`

- 结果：
  - `val_loss = 0.3619`
  - `val_recon = 0.3600`
  - `val_shared_recon = 0.3620`
  - `val_scaled_residual = 0.0034`
  - L1 `[20, 60]`
  - L2 `[20, 23, 37]`
  - L3 `[18, 2, 3, 3, 25, 29]`
  - 验证分析中的有效帧总数：`80`

- 结论：
  - FSQ 在 `win20-step20` 下仍给出非坍缩的 code usage
  - 与旧 `win10-step10` 轮次相比，重建更弱，且验证证据基于更少的有效帧
  - 由于与新的标准数据集设置匹配，此运行应作为未来探测的活跃基线

### 结构探测：`win20-step20` 上的 `v11 private_dim=8`

- 状态：已完成
- 命令：

```bash
bash scripts/lq/run_train_x_mouth_v11_private_dim8_probe.sh
```

- 预期输出：
  - `outputs/lq_x_mouth_v11_private_dim8_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v11_private_dim8_probe_win20/analysis/summary.json`

- 结果：
  - `val_loss = 0.3627`
  - `val_recon = 0.3610`
  - `val_shared_recon = 0.3625`
  - `val_scaled_residual = 0.0026`
  - L1 `[20, 60]`
  - L2 `[0, 80, 0]`
  - L3 `[19, 1, 4, 2, 54, 0]`
  - 验证分析中的有效帧总数：`80`

- 与 `v10` 对比：
  - 更差的 `val_loss`
  - 更差的 `val_recon`
  - 更差的 `val_shared_recon`
  - 更低的 `scaled_residual`
  - 明显更差的 code usage，尤其是 L2 从 `[20, 23, 37]` 坍缩到 `[0, 80, 0]`

- 结论：
  - 在当前 `win20 + FSQ` 设置下，将 `private_dim` 从 `32` 缩小到 `8` 不是好的首要收紧策略
  - 它降低了残差幅度，但损害了重建和 code usage
  - 此探测应被判定为拒绝，不予推广

### 结构探测：`win20-step20` 上的 `v12 private_decoder_hidden_dim=16`

- 状态：已完成
- 命令：

```bash
bash scripts/lq/run_train_x_mouth_v12_private_decoder16_probe.sh
```

- 预期输出：
  - `outputs/lq_x_mouth_v12_private_decoder16_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v12_private_decoder16_probe_win20/analysis/summary.json`

- 结果：
  - `val_loss = 0.3636`
  - `val_recon = 0.3615`
  - `val_shared_recon = 0.3620`
  - `val_scaled_residual = 0.0039`
  - L1 `[20, 60]`
  - L2 `[20, 22, 38]`
  - L3 `[17, 3, 2, 3, 17, 38]`
  - 验证分析中的有效帧总数：`80`

- 与 `v10` 对比：
  - 更差的 `val_loss`
  - 更差的 `val_recon`
  - `val_shared_recon` 几乎不变
  - 更差的 `scaled_residual`
  - code usage 保持较分散，但不比基线明显更好

- 结论：
  - 将 private decoder 宽度从基线有效值 `64` 收窄到 `16` 未能改善当前 `win20 + FSQ` 设置
  - 与 `v11` 相比，此方向破坏性较小（未导致 L2 坍缩），但仍未能超越 `v10`
  - 此探测也应被判定为拒绝

### 辅助探测：`win20-step20` 上的 `v13 side_cont_weight=0.15`

- 状态：已完成
- 命令：

```bash
bash scripts/lq/run_train_x_mouth_v13_side_cont_probe.sh
```

- 预期输出：
  - `outputs/lq_x_mouth_v13_side_cont_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v13_side_cont_probe_win20/analysis/summary.json`

- 重要说明：
  - `side_disc_weight=0.0`，因此离散 side 监督未优化
  - 但传入 `side_labels` 后，模型仍计算并报告 `side_disc` 指标供检查
  - 由于此探测的 total loss 中包含了 `side_cont`，total `val_loss` 不再与基线目标直接可比

- 结果：
  - `val_recon = 0.3623`
  - `val_shared_recon = 0.3631`
  - `val_scaled_residual = 0.0018`
  - `val_side_cont = 1.0522`
  - L1 `[8, 72]`
  - L2 `[46, 34, 0]`
  - L3 `[31, 28, 9, 11, 1, 0]`
  - 验证分析中的有效帧总数：`80`

- 与 `v10` 对比：
  - 重建更差
  - shared 重建更差
  - 缩放残差幅度更低
  - code usage 保持分散，但显著偏移，未能产生更清晰的 shared-motion 改善

- 结论：
  - 在当前 `win20 + FSQ` 基线上重新引入 `0.15` 的连续 side 监督未能改善 shared-motion 重建
  - 它改变了 code 分配并降低了残差幅度，但 shared-motion 行为并未更好，重建略差
  - 此探测目前应被判定为拒绝

### 结构探测：`win20-step20` 上的 `v14 basis_orthogonalization=level_qr`

- 状态：已完成
- 命令：

```bash
bash scripts/lq/run_train_x_mouth_v14_level_qr_probe.sh
```

- 预期输出：
  - `outputs/lq_x_mouth_v14_level_qr_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v14_level_qr_probe_win20/analysis/summary.json`

- 重要说明：
  - 此探测仅在每个 level 内部强制严格正交性
  - 报告的 `orth` 指标仍是全 11 个 basis 的全局软惩罚，因此不为零，因为跨 level 相似性仍被允许并被惩罚

- 结果：
  - `val_recon = 0.3574`
  - `val_shared_recon = 0.3604`
  - `val_scaled_residual = 0.0049`
  - L1 `[20, 60]`
  - L2 `[21, 59, 0]`
  - L3 `[19, 2, 2, 2, 5, 50]`
  - 验证分析中的有效帧总数：`80`

- 与 `v10` 对比：
  - 更好的重建
  - 更好的 shared 重建
  - 更差的残差幅度
  - 明显更集中的 code usage，尤其在 L2 和 L3

- 结论：
  - level-wise QR 是在 `win20 + FSQ` 基线上第一个改善 shared-path 重建的后续探测
  - 然而，它在 code usage 变得更集中时做到了这一点
  - 这应被视为混合结果，而非干净的新的基线
  - 最可能的下一步是保持 QR 可选，并在其上测试系数尺度控制，而非单独推广 QR

### 结构探测：`win20-step20` 上的 `v15 basis_orthogonalization=global_qr`

- 状态：已完成
- 动机：
  - `level_qr` 仅防止每个 level 内的 basis 相似性
  - 若希望所有 11 个 basis 相互不同，QR 步骤必须对整个 basis bank 应用一次

- 实现：
  - `scripts/lq/model/network.py` 现在支持 `basis_orthogonalization=global_qr`
  - `global_qr` 对展平的完整 basis bank 运行 QR，然后 reshape 回 `(sum(levels), H, W)`
  - 这意味着跨 level 的 basis 相似性不再被结构化投影本身允许

- 探测入口：

```bash
bash scripts/lq/run_train_x_mouth_v15_global_qr_probe.sh
```

- 预期输出：
  - `outputs/lq_x_mouth_v15_global_qr_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v15_global_qr_probe_win20/analysis/summary.json`

- 结果：
  - `val_loss = 0.3588`
  - `val_recon = 0.3572`
  - `val_shared_recon = 0.3597`
  - `val_scaled_residual = 0.0041`
  - L1 `[18, 62]`
  - L2 `[19, 5, 56]`
  - L3 `[58, 22, 0, 0, 0, 0]`
  - 验证分析中的有效帧总数：`80`

- 与 `v10` 对比：
  - 更好的总重建
  - 更好的 shared 重建
  - 略大的残差幅度
  - 高层 code usage 明显更集中，尤其在 L3

- 与 `v14` 对比：
  - 略好的总重建
  - 略好的 shared 重建
  - 略低的残差幅度
  - code usage 在更高层级更集中

- 结论：
  - 完整 bank QR 成功执行了更强力的结构先验：任何两个 basis（跨 level 也不例外）不应相似
  - 然而，当前 decoder/系数路径的响应是使用更少的离散码，而非将 usage 分散到更distinct的 basis bank 上
  - 这应被视为结构确认，而非新基线

## 决策

- 活跃基线：`outputs/lq_x_mouth_v10_fsq_probe_win20`
- 拒绝的探测：`outputs/lq_x_mouth_v11_private_dim8_probe_win20`
- 拒绝的探测：`outputs/lq_x_mouth_v12_private_decoder16_probe_win20`
- 拒绝的探测：`outputs/lq_x_mouth_v13_side_cont_probe_win20`
- 混合探测：`outputs/lq_x_mouth_v14_level_qr_probe_win20`
- 混合探测：`outputs/lq_x_mouth_v15_global_qr_probe_win20`
- 建议下一步：
  1. 若在优化当前 `x + mouth` 设置，停止收紧 basis 正交性，转而在 shared 系数 / decoder 方面努力，使模型实际能够使用更多离散 bank
  2. 否则，在 `mode=y` 上验证 FSQ 趋势
  3. 并行地，重新审视数据集/样本效率策略，因为验证集仍有只有 `80` 个有效帧

### 结构变化：添加 post-QR basis 稀疏损失

- 状态：作为 30 轮探测完成
- 动机：
  - 在 `global_qr` 之后，所有 basis 全局正交，但它们仍是空间密集的
  - 在结构化 basis bank 本身上添加显式稀疏先验

- 实现：
  - `scripts/lq/model/network.py` 现在暴露 `basis_l1`
  - `basis_l1` 在 `get_structured_basis()` 返回的结构化 basis 上计算，因此顺序为：
    1. 强制矩阵结构
    2. 在配置时应用 QR 投影
    3. 在结果 basis bank 上应用 L1 稀疏损失
  - QR 辅助函数也简化为 EDTalk 风格的可微分 QR 投影（无额外符号校正步骤）

- 训练钩子：
  - `scripts/lq/train.py` 现在支持 `basis_l1_weight`
  - 探测入口：

```bash
bash scripts/lq/run_train_x_mouth_v16_global_qr_basis_l1_probe.sh
```

- 冒烟检查：
  - `basis_l1 = 0.00639`
  - `orth ~= 0`
  - `batch_size=64`、`group_size=4` 上前向+反向通过

- 30 轮结果：
  - 运行：`outputs/lq_x_mouth_v16_global_qr_basis_l1_probe_win20_e30`
  - `val_loss = 0.3478`
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3580`
  - `val_scaled_residual = 0.0353`
  - `val_basis_l1 = 0.00266`
  - L1 `[56, 24]`
  - L2 `[18, 7, 55]`
  - L3 `[0, 0, 0, 0, 7, 73]`
  - 验证分析中的有效帧总数：`80`

- 与 `v15` 对比：
  - 好得多的总重建
  - 略好的 shared 重建
  - 明显更大的 private 残差贡献
  - 更高层的 code usage 更加集中

- 结论：
  - post-QR basis 稀疏性在数值上有效：basis L1 从冒烟检查 `0.00639` 降到验证 `0.00266`
  - 然而，模型通过将更多负载推到 private 残差分支来为稀疏性付出代价
  - 尽管 headline 重建改善，这不是干净的可解释性收获

### 结构探测：`win20-step20` 上的 `v17 residual_fsq + global_qr + basis_l1`

- 状态：已完成
- 动机：
  - 用残差量化栈替换单个 FSQ block
  - 将 stage 1/2/3 与现有 basis 分区 `(2, 3, 6)` 对齐
  - 保持 `global_qr` 和 basis 稀疏性启用

- 实现：
  - 官方的 `FSQ` blocks 以残差形式堆叠在 `scripts/lq/model/network.py` 内
  - stage 1 使用 `levels=[2]`，stage 2 使用 `levels=[3]`，stage 3 使用 `levels=[6]`
  - 每个 stage 量化剩余残差并馈送到匹配的 basis 分支

- 探测入口：

```bash
bash scripts/lq/run_train_x_mouth_v17_residual_fsq_basis_l1_probe.sh
```

- 结果：
  - `val_loss = 0.3291`
  - `val_recon = 0.3109`
  - `val_shared_recon = 0.3423`
  - `val_scaled_residual = 0.0393`
  - `val_basis_l1 = 0.00239`
  - L1 `[22, 58]`
  - L2 `[50, 8, 22]`
  - L3 `[55, 2, 1, 3, 13, 6]`
  - 验证分析中的有效帧总数：`80`

- 与 `v16` 对比：
  - 好得多的总重建
  - 差得多的 shared 重建
  - 略大的 private 残差贡献
  - 明显更健康的高层 code usage，尤其在 L3

- 结论：
  - residual FSQ 相对于坍缩的 `v16` 稀疏 basis 运行改善了离散码利用率
  - 然而，它仍未解决主要结构问题：模型通过依赖 private 残差分支来不断改善总重建，而 shared 重建变差
  - 将此视为有用的结构探测，而非新的可解释性基线

### 结构探测：`win20-step20` 上的 `v18 residual_fsq + sparse_shared_mixing + shared_recon_loss`

- 状态：已完成
- 动机：
  - 保持 residual FSQ
  - 用 anchor 引导的稀疏混合增加 shared-path 表达性
  - 添加对 `shared_recon` 的直接优化压力

- 实现：
  - `shared_basis_soft_mixing=True`
  - `shared_basis_anchor_bias=2.0`
  - `shared_basis_topk=2`
  - `shared_recon_weight=1.0`

- 探测入口：

```bash
bash scripts/lq/run_train_x_mouth_v18_residual_fsq_sparse_shared_probe.sh
```

- 结果：
  - `val_recon = 0.3150`
  - `val_shared_recon = 0.3469`
  - `val_scaled_residual = 0.0394`
  - `val_basis_l1 = 0.00251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 8, 12]`
  - 验证分析中的有效帧总数：`80`

- 重要说明：
  - 由于此运行向 total 目标添加了 `shared_recon_weight=1.0`，`val_loss` 与旧探测不完全可比

- 与 `v17` 对比：
  - 略差的总重建
  - 明显更好的 shared 重建
  - private 残差幅度仍然高，大致不变
  - code usage 在 L3 保持多码，但比 `v17` 略集中

- 结论：
  - 这验证了结构方向：增加 shared-path 容量并直接监督 `shared_recon` 确实将模型从最严重的 shared-path 坍缩中拉回
  - 然而，它还不足以减少对 private 残差分支的依赖
  - 下一步应保持此 shared-path 设计，并在此刻明确收紧 private 分支，因为 shared 分支已有更多容量

### 结构探测：`win20-step20` 上的 `v19 v18 + 更严格的 private 残差上限`

- 状态：已完成
- 动机：
  - 保持 `v18` 改进的 shared-path 设计
  - 收紧 private 残差分支，测试模型是否能在更小的 private 修正下保持更好的 shared 重建

- 实现：
  - 与 `v18` 相同，除了 `private_residual_max_l1=0.5`

- 探测入口：

```bash
bash scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh
```

- 结果：
  - `val_recon = 0.3275`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0214`
  - `val_basis_l1 = 0.00252`
  - L1 `[22, 58]`
  - L2 `[58, 0, 22]`
  - L3 `[57, 1, 1, 1, 10, 10]`
  - 验证分析中的有效帧总数：`80`

- 重要说明：
  - 由于此运行保持 `shared_recon_weight=1.0`，`val_loss` 仍与旧探测不完全可比

- 与 `v18` 对比：
  - 更差的总重建
  - 略差的 shared 重建
  - 低得多的 private 残差贡献
  - 类似的 code usage 模式，L2 集中度略强

- 与 `v17` 对比：
  - 更差的总重建
  - 明显更好的 shared 重建
  - 低得多的 private 残差贡献

- 结论：
  - 这是当前阶段第一个有意义地改善可解释性权衡的探测
  - shared path 保持比 `v17` 强得多，而 private 分支不再以与 `v17`/`v18` 相同的幅度占主导
  - 下一步应在该 regime 附近做局部搜索，而非回到无约束 private 残差

### 结构探测：`win20-step20` 上的 `v20 v19 + 上限=0.4`

- 状态：已完成
- 动机：
  - 测试若 private 残差上限进一步收紧，`v19` regime 是否仍有改善
  - 保持 shared-path 设计不变，以便比较隔离上限强度

- 实现：
  - 与 `v19` 相同，除了 `private_residual_max_l1=0.4`

- 探测入口：

```bash
bash scripts/lq/run_train_x_mouth_v20_residual_fsq_sparse_shared_privatecap04_probe.sh
```

- 结果：
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3448`
  - `val_scaled_residual = 0.0171`
  - `val_basis_l1 = 0.00252`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 17, 3]`
  - 验证分析中的有效帧总数：`80`

- 与 `v19` 对比：
  - 更差的总重建
  - 略差的 shared 重建
  - 更低的 private 残差贡献
  - 高层 usage 更多地集中到一个后期码

- 结论：
  - `cap=0.4` 是更严格的 private 变体的可行方案，但不是干净升级
  - 它改善了 private-suppression 指标，但牺牲了一点 shared 质量和更高层 code 分散度

### 结构探测：`win20-step20` 上的 `v21 v19 + 上限=0.6`

- 状态：已完成
- 动机：
  - 测试 `v19` 局部权衡窗口的另一侧
  - 检查略宽松的 private 上限是否反而能改善 shared 行为，而不仅仅是普通重建

- 实现：
  - 与 `v19` 相同，除了 `private_residual_max_l1=0.6`

- 探测入口：

```bash
bash scripts/lq/run_train_x_mouth_v21_residual_fsq_sparse_shared_privatecap06_probe.sh
```

- 结果：
  - `val_recon = 0.3245`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0251`
  - `val_basis_l1 = 0.00251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 11, 9]`
  - 验证分析中的有效帧总数：`80`

- 与 `v19` 对比：
  - 更好的总重建
  - 基本上 shared 重建不变
  - 明显更大的 private 残差贡献
  - L3 分散度可接受，但收益主要通过 private 分支而非更强的 shared path 获得

- 结论：
  - `cap=0.6` 对当前可解释性目标来说太宽松
  - 它不应取代 `v19` 作为当前基线

### 局部 sweep 结论：`private_residual_max_l1 in {0.4, 0.5, 0.6}`

- `v19 (cap=0.5)` 仍是最安全的可解释性基线
- `v20 (cap=0.4)` 是当优先降低 `scaled_residual` 时的更严格 private 替代方案
- `v21 (cap=0.6)` 确认放宽上限主要恢复 private 修正，而非 shared 解释能力
