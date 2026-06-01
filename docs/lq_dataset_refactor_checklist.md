# LQ 数据集重构检查清单

最后更新：2026-04-18

## 背景

当前 `scripts/lq/datasets.py` 是 LQ 解耦原型的最小数据集实现。它已经可用于冒烟测试，但尚未完整编码 `data/win20-step20/TT` 和 `data/win20-step20/IMR` 中实际研究数据的语义。

本检查清单记录了在向训练流程做更大更改前商定的重构优先级。

## 当前缺口

1. 区域定义在多个文件间重复，应统一。
2. `mode=x|y` 当前仅切换加载的文件后缀，但尚未强制执行方向特定的样本有效性逻辑。
3. `metadata.csv` 中的 `deleted_x` / `deleted_y` 在构造有效差分样本时尚未使用。
4. 数据集预处理和 action-basis 初始化尚未在一处共享所有结构假设。
5. 样本元数据对下游分析仍不足。
6. 采样仍是窗口扁平的，尚未按患者平衡或 group 平衡。

## 商定的优先级

### Phase 1：从速执行

1. 统一区域定义
   - 对 `full` 和 `mouth` 使用一个共享定义。
   - `mouth` 应遵循约定的裁剪范围 `188:307`。
   - 在数据集加载、basis 初始化和未来可视化脚本中复用相同的区域定义。

2. 明确方向特定的数据集语义
   - `mode=x` 和 `mode=y` 最终应成为真正的方向特定样本构造器，而不仅是不同的文件名。
   - 一旦添加 `deleted_x` / `deleted_y` 过滤，这尤其重要。

3. 添加 `deleted_x` / `deleted_y` 过滤
   - 源文件夹：`data/win20-step20/TT` 和 `data/win20-step20/IMR`
   - 它们的 `metadata.csv` 包含 `deleted_x` 和 `deleted_y`。
   - 后续实现打算使用的规则：
     - `mode=x`：跳过当前差分窗口被标记为 `deleted_x` 的样本
     - `mode=y`：跳过当前差分窗口被标记为 `deleted_y` 的样本
   - 此项目故意推迟，但它是下一个重要的语义修复。

4. 将数据集结构先验与 basis 初始化对齐
   - Action basis 初始化已应用：
     - 对称性强制
     - 零对角
     - 区域裁剪
   - 数据集预处理后续应支持与这些假设的可选对齐。

5. 改进每样本元数据
   - 保留或添加稳定字段，如：
     - `subject`
     - `dataset_name`
     - `window_idx`
     - `mode`
     - `sample_id`
   - 这对后续分析 code 激活和重建行为是必需的。

6. 明确标签层次
   - 主要监督：`side_label`
   - 辅助监督：`severity_label`、`dataset_label`
   - 原始元数据：`label_5class`、`score`

### Phase 2：稍后执行

1. 按患者平衡采样
   - 避免窗口多的患者主导优化。

2. 数据集平衡或 side 平衡采样
   - 若 IMR/TT 或 side 分布使训练偏差，这很有用。

3. 可选 raw-pair 输出
   - 在需要时返回 `current_matrix`、`prev_matrix` 和 `delta_matrix`。

4. 未来多分支可扩展性
   - 为后续 `xy` 或双分支输入模式留出空间。

5. 更丰富的 group 元数据
   - 若后续 loss 或分析需要，为源 / side / severity 的显式分组字段留出空间。

6. 缓存/加速
   - 仅在数据集语义稳定后再考虑。

## 推荐执行顺序

1. 提取共享区域常量/工具函数。
2. 改进数据集元数据接口。
3. 添加方向感知的 `deleted_x` / `deleted_y` 过滤。
4. 重新审视矩阵结构对齐和采样策略。

## 备注

- 目前 `deleted_x` / `deleted_y` 处理故意推迟。
- 近期焦点仍是保持当前流程可运行，同时逐步将其与实际研究语义对齐。
