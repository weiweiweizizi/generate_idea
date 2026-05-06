# DisentangleNet Label5Class Dual-Path Design

**状态**: 草稿

**日期**: 2026-05-06

---

## 1. 背景

`scripts/disentangleNet` 当前是从 `scripts/lq` 冻结出来的 `v31` 训练与分析闭包，训练主线默认依赖一条 3 类 laterality 监督链：

- 数据层从 `label_5class` 派生出 `side_label`
- 训练层主要消费 `batch["side_label"]`
- 模型层的 `side_classifier` / `group_side_classifier` 默认按 3 类构建
- 分析层默认围绕 `side_label`、`side_label_name` 做 probe、聚合和可解释性输出

这套路径在当前阶段可用，但存在两个问题：

1. 训练主标签语义被固定成 `side`
2. 后续如果继续尝试更细粒度标签，现有结构缺少平滑迁移与回退能力

本轮需求不是简单把 3 改 5，而是做一次可回退的结构升级：

1. 保留现有 `side` 路径，便于回退与历史结果对比
2. 新增并默认切向 `label_5class` 的 5 类监督路径
3. 让分析链支持“主标签可切换、兼容标签保留”
4. 在不改后续接口尺寸的前提下，把 shared CNN trunk 再加深一层

---

## 2. 目标

本设计的目标有四个：

1. 在 `disentangleNet` 训练闭包中新增 `label_5class` 主监督能力，同时保留 `side` 监督路径
2. 让训练与分析都能通过显式配置选择标签模式，而不是把 `side` 写死成唯一语义
3. 在 shared CNN trunk 中，于每次残差下采样前新增一个 `BasicBlock`，但不改变后续 branch 的输入维度与分析接口
4. 保证旧 checkpoint、旧 `side` 逻辑和现有分析脚本能继续工作，且新流程可独立使用 `label_5class`

成功标准：

1. `target_label_mode=label5class` 时，训练可正常起跑、反传并保存 checkpoint
2. `target_label_mode=side` 时，旧训练流程不被破坏
3. 新 checkpoint 能被核心分析脚本读取
4. 导出表中保留 `label_5class` 与派生的 `side_label / side_label_name`
5. trunk 加深后，`DistNet` 下游 branch 形状和 grouped forward 接口不需要额外改协议

---

## 3. 总体方案

### 3.1 双路径并存

本轮不做“彻底替换 side 分支”，而是引入双路径并存：

- `side` 路径继续存在，负责 3 类监督与回退能力
- `label5class` 路径新增为 5 类监督与默认新主线

统一通过显式配置控制：

- `target_label_mode="side"`
- `target_label_mode="label5class"`
- `target_label_mode="both"`

其中推荐的新默认值是：

- `target_label_mode="label5class"`

这样既能支持后续主线迁移，也不会丢失旧路径。

### 3.2 主标签与兼容标签解耦

数据层继续同时携带：

- `label_5class`
- `side_label`
- `side_label_name`
- `score`

区别只在于：

- 训练主监督按 `target_label_mode` 选择
- 分析默认视角切到 `label_5class`
- `side_label` 退化为派生兼容标签，而不是唯一主标签

### 3.3 trunk 加深但接口不变

shared CNN trunk 当前是：

- `initial_conv(stride=2)`
- `layer1(stride=2)`
- `layer2(stride=2)`
- `layer3(stride=1)`

本轮按已确认的技术假设，仅在两次残差下采样前各加一个不下采样 `BasicBlock`：

- `initial_conv`
- `pre_layer1_block`，`8 -> 8`, `stride=1`
- `layer1`，`8 -> 16`, `stride=2`
- `pre_layer2_block`，`16 -> 16`, `stride=1`
- `layer2`，`16 -> hidden_dim`, `stride=2`
- `layer3`，`hidden_dim -> hidden_dim`, `stride=1`

这里**不改** `initial_conv`，也不额外新增第三个预块。原因是：

1. 用户明确认可这条更稳的假设
2. 后续分支的输入通道和特征图尺寸都能保持稳定
3. 结构风险主要集中在 trunk 深度，而不是 stem 重写

---

## 4. 数据与配置设计

### 4.1 数据层

`scripts/disentangleNet/data/samples.py` 已经同时构造：

- `side_label = create_side_label(label_5class)`
- `label_5class`

因此本轮数据层不需要推翻现有 sample contract。重点是明确约束：

1. `label_5class` 始终视为原始标签
2. `side_label` 始终视为从 `label_5class` 派生出的兼容标签
3. 新训练和新分析不能再假设 `side_label` 是唯一 source of truth

### 4.2 配置层

`scripts/disentangleNet/train.py` 和 `scripts/disentangleNet/training/config.py` 需要承担新的标签模式配置责任。

新增配置建议：

- `target_label_mode`
- `num_side_classes`
- `num_label5_classes`
- `side_loss_weight`
- `label5_loss_weight`
- `group_side_loss_weight`
- `group_label5_loss_weight`

约束规则：

1. `target_label_mode="side"` 时，仅 side 监督必需
2. `target_label_mode="label5class"` 时，仅 label5class 监督必需
3. `target_label_mode="both"` 时，两套监督都启用，且权重必须显式可见

配置写入 checkpoint，供分析脚本读取，不允许分析脚本依赖硬编码常量去猜测类别数和标签语义。

---

## 5. 模型设计

### 5.1 trunk 结构改动

`scripts/disentangleNet/model/encoder.py` 负责 trunk 搭建。这里的设计目标是：

1. 增加残差深度
2. 不改变 `DistNet` 后续 branch 接口
3. 尽量减少对分析和 checkpoint 结构的额外扰动

因此 `build_motion_encoder(...)` 的返回协议仍然是“一组固定 stage 模块”，但内部会多出两个 pre-downsample block。

`scripts/disentangleNet/model/distnet.py` 只需要按新顺序执行 trunk，不需要改 free/side/private adapter 的输入通道定义。

### 5.2 分类头改动

当前模型头部主要围绕：

- `side_classifier`
- `group_side_classifier`

本轮需要扩成并行双头：

- `side_classifier`
- `group_side_classifier`
- `label5_classifier`
- `group_label5_classifier`

关键原则：

1. `side_semantic_enabled` 与 side basis 这条语义路径先保留原名，不在这一轮强行重命名
2. 分类头的语义必须显式区分 `side` 与 `label5class`
3. forward 输出要同时暴露两套 logits，避免训练和分析各自重复推导

### 5.3 forward 输出协议

`DistNet.forward(...)` 输出里应增加清晰命名的字段，例如：

- `side_logits`
- `group_side_logits`
- `label5_logits`
- `group_label5_logits`

同时保留：

- `side_path_usage`
- `side_path_representation`
- `group_pooled_side_rep`

因为这些张量仍然是分析 side semantics 和 side basis 行为所需的核心中间量。

本轮不要求把 `side_path_representation` 重命名为 `label5_path_representation`。语义 basis 路径仍然是共享的“语义分支”，监督标签变得可配置即可。

---

## 6. 训练与损失设计

`scripts/disentangleNet/training/losses.py` 当前默认把：

- `batch["side_label"]`

作为主要监督目标。

本轮改造后，loss 层要支持三种模式：

### 6.1 `side` 模式

保持旧行为：

- frame/group side 分类都可用
- `label_5class` 仅作为附带元信息存在

### 6.2 `label5class` 模式

新主线：

- 主监督读取 `batch["label_5class"]`
- group pooling 后走 `group_label5_classifier`
- side 可不参与 loss，或仅保留为零权重兼容输出

### 6.3 `both` 模式

双头同时训练：

- side 负责粗粒度 laterality
- label5class 负责细粒度 5 类监督

第一版实现重点不是寻找最优权重，而是保证结构闭包可用、便于对照和回退。

因此 `both` 模式可保守实现为：

- side loss 一套
- label5class loss 一套
- 各自独立加权求和

不在这一轮引入更复杂的蒸馏、层级标签一致性或多任务平衡机制。

---

## 7. 分析链迁移设计

### 7.1 默认主视角切到 `label_5class`

分析脚本当前普遍兼容 `label_5class` 字段，但默认论述仍偏向 `side_label`。本轮需要明确：

1. 新 checkpoint 默认按 `label_5class` 解释
2. 旧 checkpoint 默认按 `side` 解释
3. 分析脚本允许显式参数覆盖默认主视角

首批需要同步的脚本包括：

- `scripts/disentangleNet/analysis/analyze_checkpoint.py`
- `scripts/disentangleNet/analysis/analyze_side_interpretability.py`
- `scripts/disentangleNet/analysis/analyze_kfold_report.py`
- `scripts/disentangleNet/analysis/export_window_basis_activations.py`
- `scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py`
- `scripts/disentangleNet/analysis/analyze_patient_tsne.py`

### 7.2 导出表保持兼容字段

即使主视角切到 `label_5class`，导出表仍必须保留：

- `side_label`
- `side_label_name`

理由是：

1. 它们可以稳定从 `label_5class` 派生
2. 后续用户仍可能需要 laterality 视角快速聚合
3. 这能让新旧分析口径并行存在，而不是强迫一次性完全迁移

### 7.3 checkpoint 识别

分析脚本不能依赖“路径名里像不像 side3”来判断 checkpoint 语义。

应以 checkpoint config 为准，至少写入：

- `target_label_mode`
- `num_side_classes`
- `num_label5_classes`

读取规则：

1. 若缺失这些新字段，则按旧 side checkpoint 处理
2. 若存在这些新字段，则按其声明解释

这样旧 checkpoint 不需要重存一遍。

---

## 8. 验证方案

### 8.1 结构冒烟

先做不依赖长训练的结构验证：

1. 单 batch forward 成功
2. logits 形状与类别数匹配
3. grouped input 下 frame/group supervision 路径都能走通

### 8.2 小规模训练冒烟

以最小 epoch 或小批次配置验证：

1. `target_label_mode=label5class` 可训练
2. `target_label_mode=side` 可回退
3. `target_label_mode=both` 至少能完成一次完整 train/val loop

### 8.3 分析链冒烟

至少验证：

1. `analyze_checkpoint.py` 能读取新 checkpoint
2. `export_window_basis_activations.py` 能导出包含 `label_5class` 主字段和 `side_label` 兼容字段的表
3. 至少一个患者级脚本能跑通，例如：
   - `analyze_patient_activation_patterns.py`
   - 或 `analyze_patient_tsne.py`

---

## 9. 风险与不包含内容

### 9.1 风险

1. trunk 加深后，旧 checkpoint 不能直接加载到新模型结构
2. 双头并存会让训练配置复杂化，若命名不清，容易再次出现语义混淆
3. 分析脚本若只改默认视角但未补充 checkpoint 判别，容易把旧模型误按新逻辑解释

### 9.2 不包含内容

本轮明确不包含：

1. side semantic basis 的全面重命名
2. 新的 basis 数量搜索
3. 新的 pooling 设计
4. 对历史输出目录的批量迁移
5. 对所有旧 checkpoint 的重分析重写

当前重点是：

- 建立一个可并存、可回退、可继续迭代的标签监督框架
- 在此基础上完成一次低风险 trunk 加深

