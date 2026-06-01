# LQ 训练预设配置

当前 `scripts/lq` 的标准数据集根目录：

- `data/win20-step20/IMR,data/win20-step20/TT`

重构兼容性说明：

- 第一轮将内部实现拆分到 `scripts/lq/training/`、`scripts/lq/data/` 和 `scripts/lq/model/`
- 训练入口和现有 shell 脚本 CLI 形状保持不变
- 本文档中的当前预设应与以前完全相同的方式启动

## 保守预设

当前推荐的首轮训练预设：

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`3.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_weight：`0.15`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v1.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v1
```

## 保守抗坍缩预设

这是在 v2 显示 `weight_decay + optimize_values=False` 单独未能缓解坍缩之后的当前下一步比较预设。

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.15`
- side_disc_weight：`0.0`
- weight_decay：`0.001`
- lq_commitment_loss_weight：`1.0`
- lq_quantization_loss_weight：`1.0`
- lq_optimize_values：`False`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v3_no_disc_side_stronger_lq.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v3_no_disc_side_stronger_lq
```

## 低残差诊断预设

这保持 v3 官方 LQ 设置，但约束 private 残差分支，因此 shared action bases 必须更直接地解释矩阵。

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.15`
- side_disc_weight：`0.0`
- private_residual_weight：`0.05`
- weight_decay：`0.001`
- lq_commitment_loss_weight：`1.0`
- lq_quantization_loss_weight：`1.0`
- lq_optimize_values：`False`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v4_low_residual.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v4_low_residual
```

## 低残差有上限预设

这是在 v4 基础上扩展，限制每样本 private 残差均值绝对值，防止模型通过膨胀残差幅度同时保持小残差权重来恢复相同的逃逸路径。

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.15`
- side_disc_weight：`0.0`
- private_residual_weight：`0.05`
- private_residual_max_l1：`1.0`
- weight_decay：`0.001`
- lq_commitment_loss_weight：`1.0`
- lq_quantization_loss_weight：`1.0`
- lq_optimize_values：`False`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v5_low_residual_capped.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v5_low_residual_capped
```

## 无侧别探测预设

这保持 v5 有上限残差设置，但完全禁用 side 监督。它是检验 side 标签是否是 shared 离散码坍缩的主要力量的探测。

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.0`
- side_disc_weight：`0.0`
- private_residual_weight：`0.05`
- private_residual_max_l1：`1.0`
- weight_decay：`0.001`
- lq_commitment_loss_weight：`1.0`
- lq_quantization_loss_weight：`1.0`
- lq_optimize_values：`False`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v6_no_side_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v6_no_side_probe
```

## 共享瓶颈探测预设

这保持 v6 无侧别、有上限残差设置，但将 shared 潜在从 `32` 缩小到 `8`。它测试 shared 量化路径坍缩是否因为其连续预量化表示仍然过于表达。

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- hidden_dim：`32`
- shared_dim：`8`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.0`
- side_disc_weight：`0.0`
- private_residual_weight：`0.05`
- private_residual_max_l1：`1.0`
- weight_decay：`0.001`
- lq_commitment_loss_weight：`1.0`
- lq_quantization_loss_weight：`1.0`
- lq_optimize_values：`False`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v7_shared_bottleneck.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v7_shared_bottleneck
```

## Pool-2 探测预设

这保持 v6 无侧别、有上限残差基线，但将最终 `1x1` 自适应平均池替换为 `2x2`，使 shared path 看到粗粒度空间布局，而非每个通道仅一个全局均值。

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- hidden_dim：`32`
- pool_size：`2`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.0`
- side_disc_weight：`0.0`
- private_residual_weight：`0.05`
- private_residual_max_l1：`1.0`
- weight_decay：`0.001`
- lq_commitment_loss_weight：`1.0`
- lq_quantization_loss_weight：`1.0`
- lq_optimize_values：`False`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v8_pool2_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v8_pool2_probe
```

## 软 Basis 探测预设

这保持 v6 无侧别、有上限残差基线，但将 shared 重建从硬单 basis 选择升级为逐 level 软混合，使用离散选中码作为 anchor bias。

- 模式：`x`
- 区域：`mouth`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- hidden_dim：`32`
- pool_size：`1`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.0`
- side_disc_weight：`0.0`
- private_residual_weight：`0.05`
- private_residual_max_l1：`1.0`
- shared_basis_soft_mixing：`True`
- shared_basis_anchor_bias：`1.0`
- weight_decay：`0.001`
- lq_commitment_loss_weight：`1.0`
- lq_quantization_loss_weight：`1.0`
- lq_optimize_values：`False`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v9_soft_basis_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v9_soft_basis_probe
```

## FSQ 探测预设

这是当前标准 FSQ 基线预设。它保持 v6 无侧别、有上限残差结构，并用官方 `FSQ` 替换 `LatentQuantize`，更好地匹配现有的 `levels=(2,3,6)` 因式分解。

`win20-step20` 上观察到的结果：

- `val_loss = 0.3619`
- `val_recon = 0.3600`
- `val_shared_recon = 0.3620`
- L1 `[20, 60]`
- L2 `[20, 23, 37]`
- L3 `[18, 2, 3, 3, 25, 29]`

- 模式：`x`
- 区域：`mouth`
- data_roots：`data/win20-step20/IMR,data/win20-step20/TT`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- hidden_dim：`32`
- pool_size：`1`
- quantizer_type：`fsq`
- fsq_preserve_symmetry：`True`
- use_dataset_aux：`False`
- recon_weight：`1.0`
- lq_weight：`10.0`
- orth_weight：`0.1`
- residual_weight：`0.02`
- side_cont_weight：`0.0`
- side_disc_weight：`0.0`
- private_residual_weight：`0.05`
- private_residual_max_l1：`1.0`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v10_fsq_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v10_fsq_probe_win20
```

## FSQ Private-Dim 探测

这是在 `win20-step20` 基线上第一个 FSQ 时代结构探测。相对标准 FSQ 基线仅改变一个变量：`private_dim` 从 `32` 降到 `8`。

观察到的结果：

- `val_loss = 0.3627`
- `val_recon = 0.3610`
- `val_shared_recon = 0.3625`
- `val_scaled_residual = 0.0026`
- L1 `[20, 60]`
- L2 `[0, 80, 0]`
- L3 `[19, 1, 4, 2, 54, 0]`

决策：

- 拒绝作为下一基线
- 它降低了残差幅度，但使重建略差且 L2 usage 坍缩

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v11_private_dim8_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v11_private_dim8_probe_win20
```

## FSQ Private-Decoder 探测

这是在 `win20-step20` 基线上的第二个 FSQ 时代结构探测。保持 `private_dim=32`，仅将 private decoder 隐藏宽度从基线有效值 `64` 降到 `16`。

观察到的结果：

- `val_loss = 0.3636`
- `val_recon = 0.3615`
- `val_shared_recon = 0.3620`
- `val_scaled_residual = 0.0039`
- L1 `[20, 60]`
- L2 `[20, 22, 38]`
- L3 `[17, 3, 2, 3, 17, 38]`

决策：

- 拒绝作为下一基线
- 比 `private_dim=8` 破坏性小，但仍未能超越 `v10`

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v12_private_decoder16_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v12_private_decoder16_probe_win20
```

## FSQ Side-Cont 探测

这是在 `win20-step20` FSQ 基线上的第一个辅助监督探测。保持基线结构不变，仅启用连续 side 监督 `side_cont_weight=0.15`。

观察到的结果：

- `val_recon = 0.3623`
- `val_shared_recon = 0.3631`
- `val_scaled_residual = 0.0018`
- `val_side_cont = 1.0522`
- L1 `[8, 72]`
- L2 `[46, 34, 0]`
- L3 `[31, 28, 9, 11, 1, 0]`

决策：

- 拒绝作为下一基线
- 它改变了 code 分配并降低了残差大小，但未能改善 shared-motion 重建

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v13_side_cont_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v13_side_cont_probe_win20
```

## FSQ Level-QR 探测

此探测保持标准 `win20-step20` FSQ 基线结构，但用严格 QR 正交化替换当前逐 basis 归一化（在每个 level 内部）。

观察到的结果：

- `val_recon = 0.3574`
- `val_shared_recon = 0.3604`
- `val_scaled_residual = 0.0049`
- L1 `[20, 60]`
- L2 `[21, 59, 0]`
- L3 `[19, 2, 2, 2, 5, 50]`

决策：

- 混合结果，不直接推广
- 重建改善，但 code usage 变得更集中

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v14_level_qr_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v14_level_qr_probe_win20
```

Global-QR 后续：

```bash
bash scripts/lq/run_train_x_mouth_v15_global_qr_probe.sh
```

观察到的结果：

- `val_loss = 0.3588`
- `val_recon = 0.3572`
- `val_shared_recon = 0.3597`
- `val_scaled_residual = 0.0041`
- L1 `[18, 62]`
- L2 `[19, 5, 56]`
- L3 `[58, 22, 0, 0, 0, 0]`

决策：

- 不作为新基线推广
- 重建进一步改善，但 code usage 比 level-wise QR 探测更集中

预期输出目录：

```bash
outputs/lq_x_mouth_v15_global_qr_probe_win20
```

## FSQ Global-QR + Basis-L1 探测

此探测保持 `v15 global_qr` 结构，并在 QR 投影后对结构化 basis bank 添加 L1 稀疏惩罚。

- 模式：`x`
- 区域：`mouth`
- data_roots：`data/win20-step20/IMR,data/win20-step20/TT`
- epochs：`15`
- batch_size：`64`
- group_size：`4`
- quantizer_type：`fsq`
- basis_orthogonalization：`global_qr`
- basis_l1_weight：`1.0`

冒烟检查观察：

- `basis_l1 = 0.00639`
- `orth ~= 0`
- `batch_size=64`、`group_size=4` 上前向+反向通过

30 轮观察到的结果：

- `val_loss = 0.3478`
- `val_recon = 0.3310`
- `val_shared_recon = 0.3580`
- `val_scaled_residual = 0.0353`
- `val_basis_l1 = 0.00266`
- L1 `[56, 24]`
- L2 `[18, 7, 55]`
- L3 `[0, 0, 0, 0, 7, 73]`

决策：

- 不作为可解释性基线推广
- 稀疏性先验在数值上有效，但模型通过明显更大的 private 残差分支来补偿

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v16_global_qr_basis_l1_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v16_global_qr_basis_l1_probe_win20
```

## Residual-FSQ + Global-QR + Basis-L1 探测

此探测在保持 `global_qr` 和 basis 稀疏性启用的同时，用残差 FSQ 栈替换单个 FSQ block。

- 模式：`x`
- 区域：`mouth`
- data_roots：`data/win20-step20/IMR,data/win20-step20/TT`
- epochs：`50`
- batch_size：`64`
- group_size：`4`
- quantizer_type：`residual_fsq`
- basis_orthogonalization：`global_qr`
- basis_l1_weight：`1.0`

观察到的结果：

- `val_loss = 0.3291`
- `val_recon = 0.3109`
- `val_shared_recon = 0.3423`
- `val_scaled_residual = 0.0393`
- `val_basis_l1 = 0.00239`
- L1 `[22, 58]`
- L2 `[50, 8, 22]`
- L3 `[55, 2, 1, 3, 13, 6]`

决策：

- 不作为可解释性基线推广
- residual FSQ 相对于 `v16` 改善了高层 code 分散度，但 shared 重建大幅变差且残差分支仍占主导

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v17_residual_fsq_basis_l1_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v17_residual_fsq_basis_l1_probe_win20_e50
```

## Residual-FSQ + Sparse-Shared 探测

此探测保持 residual FSQ 和稀疏 bases，然后用 anchor 引导的稀疏混合增加 shared-path 容量并添加对 `shared_recon` 的直接监督。

- 模式：`x`
- 区域：`mouth`
- data_roots：`data/win20-step20/IMR,data/win20-step20/TT`
- epochs：`50`
- batch_size：`64`
- group_size：`4`
- quantizer_type：`residual_fsq`
- shared_basis_soft_mixing：`True`
- shared_basis_anchor_bias：`2.0`
- shared_basis_topk：`2`
- shared_recon_weight：`1.0`
- basis_orthogonalization：`global_qr`
- basis_l1_weight：`1.0`

观察到的结果：

- `val_recon = 0.3150`
- `val_shared_recon = 0.3469`
- `val_scaled_residual = 0.0394`
- `val_basis_l1 = 0.00251`
- L1 `[22, 58]`
- L2 `[57, 1, 22]`
- L3 `[57, 1, 0, 2, 8, 12]`

决策：

- 有前途的方向，但还不是可解释性基线
- 相对于 `v17`，shared 重建明显改善，但 private 残差仍然过大

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v18_residual_fsq_sparse_shared_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v18_residual_fsq_sparse_shared_probe_win20_e50
```

## Residual-FSQ + Sparse-Shared + 更严格 Private 探测

此探测保持 `v18` shared-path 设计并收紧 private 残差上限以改善可解释性权衡。

- 模式：`x`
- 区域：`mouth`
- data_roots：`data/win20-step20/IMR,data/win20-step20/TT`
- epochs：`50`
- batch_size：`64`
- group_size：`4`
- quantizer_type：`residual_fsq`
- shared_basis_soft_mixing：`True`
- shared_basis_anchor_bias：`2.0`
- shared_basis_topk：`2`
- shared_recon_weight：`1.0`
- basis_orthogonalization：`global_qr`
- basis_l1_weight：`1.0`
- private_residual_max_l1：`0.5`

观察到的结果：

- `val_recon = 0.3275`
- `val_shared_recon = 0.3446`
- `val_scaled_residual = 0.0214`
- `val_basis_l1 = 0.00252`
- L1 `[22, 58]`
- L2 `[58, 0, 22]`
- L3 `[57, 1, 1, 1, 10, 10]`

决策：

- 有前途的可解释性权衡
- 相对于 `v18`，总重建更差，但 private 残差小得多，而 shared 重建仍比 `v17` 好得多
- 将此作为当前更安全的可解释性基线

运行方式：

```bash
bash scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh
```

预期输出目录：

```bash
outputs/lq_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe_win20_e50
```

## `v19` 附近局部 Private-Cap 扫描

使用与 `v19` 相同的 shared-path 结构，仅改变 `private_residual_max_l1` 运行了两个后续探测：

- `v20 cap=0.4`
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3448`
  - `val_scaled_residual = 0.0171`
  - L3 `[57, 1, 0, 2, 17, 3]`
- `v21 cap=0.6`
  - `val_recon = 0.3245`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0251`
  - L3 `[57, 1, 0, 2, 11, 9]`

解读：

- `cap=0.4` 对 private 残差抑制最好，但使更高层 usage 比 `v19` 更集中
- `cap=0.6` 改善了普通重建，但主要通过让 private 分支再次增长
- 保持 `v19 cap=0.5` 作为默认可解释性预设
- 仅在需要更严格的 private-suppression 消融时使用 `v20 cap=0.4`

## Basis Init 映射

单方向 basis init 文件位于 [`scripts/lq/init_basis`](/home/weizilin/generate_idea/scripts/lq/init_basis)。

- `x + mouth`：`scripts/lq/init_basis/basis_x.npy`
- `y + mouth`：`scripts/lq/init_basis/basis_y.npy`
- `x + full`：`scripts/lq/init_basis/basis_x_full.npy`
- `y + full`：`scripts/lq/init_basis/basis_y_full.npy`

## 等效直接命令

```bash
python scripts/lq/train.py \
  --epochs=15 \
  --batch_size=64 \
  --group_size=4 \
  --mode=x \
  --region=mouth \
  --basis_size=119 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x.npy \
  --recon_weight=1.0 \
  --lq_weight=3.0 \
  --orth_weight=0.1 \
  --residual_weight=0.02 \
  --side_weight=0.15 \
  --use_dataset_aux=False \
  --output_dir=outputs/lq_x_mouth_v1
```

## 最小变体开关

- 要运行 `y + mouth`，改变：
  - `--mode=y`
  - `--action_basis_init_path=scripts/lq/init_basis/basis_y.npy`

- 要运行 `x + full`，改变：
  - `--region=full`
  - `--basis_size=341`
  - `--action_basis_init_path=scripts/lq/init_basis/basis_x_full.npy`

- 要运行 `y + full`，改变：
  - `--mode=y`
  - `--region=full`
  - `--basis_size=341`
  - `--action_basis_init_path=scripts/lq/init_basis/basis_y_full.npy`
