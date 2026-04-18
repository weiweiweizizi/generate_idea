# LQ 单方向序列训练接入设计

**状态**: 草稿

**日期**: 2026-04-19

---

## 1. 背景与目标

当前 `scripts/lq/` 已具备以下基础：

- `datasets.py` 已支持单方向 `x` / `y` 的 signed `ΔD` 构造、`deleted_x/deleted_y` 过滤、region crop，以及固定长度序列分组。
- `network.py` 已实现 shared/private 分支、latent quantization、action basis bank、residual decoder 和基础辅助监督。
- `build_action_basis_init.py` 已可从 grouped SVD 和单患者 PC1 构造 `(sum(levels), H, W)` 的 basis 初始化张量。

但当前训练入口 `train.py` 仍基于 flat window dataset，尚未接入序列版数据，也未形成“单方向序列训练 + basis init”的完整闭环。

本设计的目标是：

1. 让 `train.py` 接入 `FacialMotionSequenceDataset`
2. 支持单方向 `x` 或 `y` 的序列输入训练
3. 保持现有 `DistNet` 的单帧解码逻辑不变
4. 使用 basis init 稳定 action basis 学习
5. 以“先跑通、先稳定、先可解释”为第一阶段目标
6. 在 `batch_size=64` 的默认实验配置下，显式校验数据载入与 batch 构造不会引发内存崩溃

本阶段**不包含**：

- 多方向联合训练（`xy/x/y` 多分支）
- 真正的时序模块（RNN / Transformer / temporal pooling）
- 严重度预测头
- blendshape 弱监督

---

## 2. 设计范围

本次设计只修改以下模块的接口层：

- `scripts/lq/train.py`
- `scripts/lq/model/network.py`

本次设计默认**不修改**以下模块的核心语义：

- `scripts/lq/datasets.py`
- `scripts/lq/utils/build_action_basis_init.py`

原因是：

- `datasets.py` 当前已经能输出序列输入及必要 mask
- `network.py` 的 shared/private/LQ/basis 主体逻辑已可复用
- 当前最小闭环是“序列输入 -> 内部逐帧前向 -> 逐帧 loss 聚合”

---

## 3. 总体方案

### 3.1 核心思路

采用“**序列输入、内部展平、逐帧重建、逐帧监督**”方案。

输入为：

- `B x T x 1 x H x W`

网络内部处理为：

1. 将 `B x T` 展平成 `B*T`
2. 复用当前单帧 `DistNet` 主体：
   - CNN encoder
   - shared / private branch
   - latent quantization
   - action basis reconstruction
   - private residual reconstruction
3. 将逐帧输出 reshape 回 `B x T x ...`

损失计算为：

- reconstruction / lq / residual / side / dataset loss 均按帧计算
- 使用不同 mask 控制哪些帧参与哪些 loss
- orthogonality loss 保持为 batch 级单个标量

### 3.2 为什么不直接做时序模块

当前第一阶段目标是：

- 跑通真实数据训练
- 验证 basis init 是否稳定
- 观察 basis / code / reconstruction 是否有可解释性

如果现在同时引入时序模块，会把问题混在一起：

- 是 dataset 接口错了
- 还是 basis init 无效
- 还是 temporal module 抢走了学习能力

因此这一阶段明确只做“序列样本承载多个帧”，不做真正时序建模。

---

## 4. 数据接口设计

### 4.1 输入数据来源

训练改为使用 `FacialMotionSequenceDataset`。

DataLoader 后单个 batch 的关键字段：

- `images`: `B x T x 1 x H x W`
- `valid_mask`: `B x T`
- `padding_mask`: `B x T`
- `side_label`: `B`
- `dataset_label`: `B`
- `severity_label`: `B`
- `sample_ids`, `sample_sources`, `window_indices`, `group_id`

### 4.2 序列语义

每个序列样本为固定长度 `group_size`：

- 优先放入 `valid` 帧
- 数量不足时，补入 `deleted_fill`
- 仍不足时，补入 `zero_pad`

这些语义不在 `network.py` 内重建，而在 `train.py` 里通过 mask 使用。

### 4.3 basis init

训练入口显式接收：

- `action_basis_init_path`

该文件由 `build_action_basis_init.py` 离线生成，按单方向分别准备：

- `mode=x` 一份
- `mode=y` 一份

第一阶段训练建议默认要求传入 basis init，不建议随机初始化直接开跑。

### 4.4 内存与 batch 约束

第一轮实验默认配置：

- `batch_size=64`
- `num_workers=0`

当前 `datasets.py` 的读取方式是：

- 在 `__getitem__()` 中按需 `np.load(...)`
- 不在 dataset 初始化阶段预加载全部矩阵

因此，第一阶段的主要内存风险不是“dataset 一次性把全量数据读入内存”，而是：

1. 一个 batch 的输入 tensor
2. DataLoader collate 后的中间张量
3. 模型前向与反向的激活占用

在 `group_size=4`、`batch_size=64` 下，单个输入 batch 的原始 float32 张量大小约为：

- `mouth (119x119)`: 约 `13.8 MiB`
- `full (341x341)`: 约 `113.6 MiB`

这只是输入本身，不含：

- reconstruction / residual / latent 等中间输出
- 反向传播激活
- CUDA 缓冲与优化器状态

因此第一阶段建议：

- 先以 `region=mouth` 作为默认训练区域
- `batch_size=64` 仅作为首选目标配置，不保证 `full` 区域一定稳定
- 在正式训练前加入一次 batch-level 内存校验

建议的校验方式：

1. 构造 train loader
2. 只取一个 batch
3. 记录 `images.shape`、mask shape、dtype
4. 估算 batch 输入 tensor 大小
5. 做一次前向与反向 smoke test
6. 若出现 OOM，则优先回退：
   - `region=mouth`
   - 减少 `group_size`
   - 减少 `batch_size`

---

## 5. 网络接口设计

### 5.1 `DistNet.forward()` 输入扩展

当前 `DistNet.forward(x, ...)` 实际只支持：

- `B x 1 x H x W`

本次扩展后支持：

- 单帧：`B x 1 x H x W`
- 序列：`B x T x 1 x H x W`

内部规则：

- 若输入为单帧，维持现有行为
- 若输入为序列，则展平为 `B*T x 1 x H x W`
- 现有单帧前向逻辑完全复用
- 输出端恢复时间维度

### 5.2 输出结构

对于逐帧输出，返回形状改为：

- `reconstructed`: `B x T x 1 x H x W`
- `action_reconstruction`: `B x T x 1 x H x W`
- `id_nuisance_residual`: `B x T x 1 x H x W`
- `shared_quantized`: `B x T x D`
- `private_z`: `B x T x Dp`
- `indices`: `B x T`
- `decoded_indices`: list of `B x T`
- `side_logits`: `B x T x C`
- `discrete_side_logits`: `B x T x C`
- `private_dataset_logits`: `B x T x C`
- `shared_dataset_logits`: `B x T x C`

对于全局量，保持标量：

- `orth_loss`

对于逐帧可聚合量，允许返回逐帧版本或在 `train.py` 内重算。推荐返回逐帧版本以便 mask 聚合。

### 5.3 标签扩展规则

序列级标签当前是：

- `side_label`: `B`
- `dataset_label`: `B`

在逐帧辅助监督时，需要在训练中扩展为：

- `B x T`

再展平为 `B*T` 与逐帧 logits 对齐。

不建议把这个逻辑放进 dataset；它属于训练语义，不属于数据存储语义。

---

## 6. Loss 与 Mask 设计

### 6.1 mask 定义

训练中派生两个主要 mask：

- `recon_mask = ~padding_mask`
- `supervision_mask = valid_mask`

语义如下：

- `valid`: 参与全部逐帧 loss
- `deleted_fill`: 参与 reconstruction / quantization / residual 相关 loss，不参与 side / dataset 辅助监督
- `zero_pad`: 不参与任何逐帧 loss

### 6.2 各项 loss 规则

#### Reconstruction loss

- 逐帧计算 `L1(reconstructed, input)`
- 只在 `recon_mask` 上聚合

#### LQ loss

- 视为逐帧量化代价
- 只在 `recon_mask` 上聚合

#### Residual L1

- 逐帧计算 private residual 的稀疏约束
- 只在 `recon_mask` 上聚合

#### Side loss

- 连续 latent 的 side 分类
- 离散 code 的 side 分类
- 只在 `supervision_mask` 上聚合

#### Dataset auxiliary loss

- private branch dataset classifier
- shared branch adversarial dataset classifier
- 只在 `supervision_mask` 上聚合

#### Orthogonality loss

- basis bank 的全局正则
- 不做逐帧 mask
- 每个 batch 只统计一次

### 6.3 为什么 `deleted_fill` 不参加辅助监督

`deleted_fill` 是真实矩阵，但被标记为弱运动窗口。

如果把它们用于：

- side 分类
- dataset 辅助分类

则有较高风险让模型在低信号帧上学习到噪声型监督。因此第一阶段保守处理：

- 用它们帮助 reconstruction
- 不用它们强化语义监督

---

## 7. `train.py` 改动清单

### 7.1 数据构造

`build_datasets()` 改为构造 `FacialMotionSequenceDataset`，并新增参数：

- `group_size`
- `apply_deleted_filter`

### 7.2 `step_model()` 改造

当前 `step_model()` 读取：

- `batch["image"]`

需改为读取：

- `batch["images"]`
- `batch["valid_mask"]`
- `batch["padding_mask"]`
- `batch["side_label"]`
- `batch["dataset_label"]`

并在内部：

1. 构造 `recon_mask` 和 `supervision_mask`
2. 调用支持序列输入的 `model(...)`
3. 对逐帧 loss 手工做 mask 聚合

### 7.3 CLI 参数

训练入口建议增加以下参数：

- `batch_size=64`
- `group_size=4`
- `apply_deleted_filter=True`
- `require_basis_init=True` 或至少在日志中显式警告

并建议增加：

- `validate_batch_memory=True`

### 7.4 checkpoint 内容

除当前保存内容外，建议附加：

- `config`
- `mode`
- `region`
- `levels`
- `group_size`
- `action_basis_init_path`

这样便于后续分析 basis 结果与训练配置的对应关系。

---

## 8. 风险与已知边界

### 8.1 这不是时序建模

虽然输入是序列，但当前版本本质上仍是：

- “逐帧编码、逐帧量化、逐帧重建”

它不会显式利用帧间依赖，只是让训练采样单元变成序列。

### 8.2 `levels[1]` 的语义耦合

当前离散 side classifier 默认绑定第二层 code：

- `self.discrete_side_classifier = nn.Embedding(self.levels[1], num_side_classes)`

这意味着：

- `levels=(2,3,6)` 不是完全自由可改
- 第二层仍被假定为 side-related level

第一阶段不改这条假设。

### 8.3 basis init 质量直接影响稳定性

如果 basis init：

- 排序错位
- mode 不匹配
- region 不匹配

则训练可能能跑，但 basis/code 解释性会明显下降。

### 8.4 loss 尺度变化

序列化后，逐帧 loss 聚合方式会改变有效样本数。

需要特别注意：

- `orth_weight` 不要被序列长度隐式放大
- 逐帧项要用 mask 归一化而不是简单 `.mean()`

### 8.5 `batch_size=64` 不是无条件安全值

`batch_size=64` 是第一轮目标配置，但是否稳定还取决于：

- `region=mouth` 还是 `region=full`
- `group_size`
- GPU 显存
- 是否开启 dataset auxiliary loss

因此，实现中应当把“batch 64 可否运行”视为待验证条件，而不是硬编码假设。

---

## 9. 第一阶段成功标准

本阶段不以分类指标为成功标准，而以训练可运行性和表示质量为标准。

成功标准如下：

1. 训练能够稳定运行，无 shape error、NaN、全零 loss
2. 能正常保存 `best.pt`
3. validation loss 呈下降或至少稳定，不是纯随机抖动
4. basis 可视化不是纯噪声，也没有完全塌缩到同一模式
5. 离散 code 使用不是极端单一
6. `valid` 帧上的 reconstruction 明显优于无意义的 pad 情况
7. `batch_size=64` 下至少能完成一次 batch-level forward/backward smoke test，不发生内存崩溃

---

## 10. 后续演进顺序

在本设计跑通后，建议按以下顺序继续：

1. 增加训练日志与分析输出
   - code usage
   - basis snapshot
   - reconstruction examples
2. 评估是否需要 patient-balanced / dataset-balanced sampling
3. 再考虑 temporal pooling 或其他时序模块
4. 再考虑 severity supervision
5. 最后再接入 blendshape 弱监督

---

## 11. 结论

现有 `datasets.py` 与 `network.py` 已足以支撑下一步“单方向序列训练 + basis init 接入”的工作，但前提是：

- 在 `train.py` 中把 flat window 训练改为 sequence-aware 训练
- 在 `network.py` 中补齐序列输入的接口适配
- 明确按帧 loss 与 mask 聚合规则

第一阶段应以“跑通训练、稳定 basis、观察 code 使用”作为唯一目标，不应同时引入时序模块、分级任务或 blendshape 监督。
