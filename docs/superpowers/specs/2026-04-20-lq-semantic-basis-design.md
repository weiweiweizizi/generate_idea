# LQ Semantic Basis Design

**状态**: 草稿

**日期**: 2026-04-20

---

## 1. 背景与目标

当前 `scripts/lq` 已经收敛到一个可运行的单方向原型：

- mode: `x`
- region: `mouth`
- data: `data/win20-step20/IMR,data/win20-step20/TT`
- 结构基线: residual FSQ + anchor-guided sparse shared mixing + shared reconstruction supervision
- 当前推荐 baseline: `v19`

本文中的“以 `v19` 为 baseline”仅表示复用 `v19` 的 backbone 结构与当前
shared/private 主干设计，不表示保持 `v19` 的训练目标完全不变。

更准确地说，第一版应视为：

- `v19 backbone + side semantic bank + group-level side supervision`

而不是：

- `v19` 原实验目标的原地小修

现阶段的核心问题已经从“能不能训起来”转向“能不能把 shared basis 变得更可解释”。

根据现有实验记录，当前已经有以下明确信号：

- 官方 `FSQ` 和 residual FSQ 能明显改善 code usage，但不能自动带来清晰语义
- 只加强 basis 正交或稀疏，会让一部分重建压力重新流向 `private residual`
- 增强 shared 路径结构并加入 `shared_recon` 后，shared branch 的承担能力确实提升
- 数据集标签 `dataset` 不应进入 shared 语义，否则会破坏跨人群泛化目标
- 当前稳定可用的临床标签是 `subject-level` 常量：
  - `side`
  - `severity`
  - `dataset`

需要特别说明的是：当前训练样本并不是整段 subject 序列，而是由
`group_size=4` 构成的 grouped sequence。因此本设计中真正参与训练与评估的监督粒度是：

- `group-level pooled representation`

其标签来源虽然是 subject-level 常量，但监督施加位置不是“整段 subject 级表示”。

本设计的目标不是要求“所有 basis 都天然可命名”，而是实现一个第一版的**语义 basis 子库**：

1. 让一小组 shared basis 明确负责 `side`
2. 让剩余 shared basis 继续负责未命名的共享运动变化
3. 为后续再接 `severity` 和 dataset disentanglement 打下结构基础

第一版只追求：

- `side` 能稳定绑定到指定的 shared semantic basis
- 总体训练行为尽量靠近当前 `v19` backbone

---

## 2. 第一版范围

本设计只做一个最小闭环，不一次性把所有临床语义都塞进模型。

### 2.1 包含内容

- 在 shared branch 内部显式拆出：
  - `B_side`
  - `B_free`
- 将 `side` 监督从当前的逐帧 shared latent / discrete code 监督，改为**group-level semantic side supervision**
- 保留当前：
  - residual FSQ
  - shared reconstruction supervision
  - private residual cap
  - basis init

### 2.2 暂不包含

- `severity` 主监督
- dataset auxiliary 的重新引入
- blendshape 主监督
- 多方向联合训练
- 大规模改写 quantizer 主干
- 完整的 concept bottleneck 架构
- 对所有 basis 做强可命名约束

`severity` 在第一版只作为**只读 probe 目标**保留接口，不进入训练主损失。

第一版应被明确理解为一个新的 probe 变体，而不是“v19 只加一个 head”的小修。

---

## 3. 为什么不直接做全量语义 basis

用户目标是尽量达到“每个 shared basis 都有独立、可命名的语义”，但当前数据条件并不支持第一步就追求这个目标。

主要原因有三点：

1. 当前视频动作模式非常单一，模型最容易先学到的是：
   - 动作强度
   - 左右侧别
   - 数据集偏差
   而不是更细的临床概念
2. `side`、`severity`、`dataset` 都是 `subject-level` 常量，不适合继续直接按帧压监督
3. 当前 shared branch 虽然已增强，但仍不足以同时承载“完整重建 + 全量可命名语义”

因此第一版采用“**小语义库 + 大自由库**”思路：

- 少量 basis 负责明确语义
- 其余 basis 保留自由表达能力

这是最接近目标 A、同时工程风险最低的版本。

---

## 4. 总体方案

### 4.1 核心思路

保持现有主干不变：

- encoder
- shared head / private head
- residual FSQ
- private residual decoder

只在 `shared_quantized` 到 `shared_reconstruction` 之间加入语义拆分：

1. `shared_quantized` 经过 side-specific mixing head，生成 `side semantic reconstruction`
2. `shared_quantized` 经过 free mixing head，生成 `free shared reconstruction`
3. 二者相加得到最终 `shared_reconstruction`
4. `private residual` 继续补充 identity / dataset / nuisance 信息

### 4.2 目标分工

- `B_side`
  - 负责承载 `side`
  - 需要尽量可命名
  - 应避免携带明显的 `dataset` 信息
- `B_free`
  - 负责其余共享运动变化
  - 不承诺可命名
  - 用于维持 shared reconstruction 能力
- `private`
  - 负责 dataset-specific / identity / nuisance
  - 明确承担 `dataset` 信息

这意味着第一版的目标不是“所有 shared 都可解释”，而是：

- **让可解释部分有明确位置**
- **让不可解释部分不再污染可解释部分**

---

## 5. 模型结构设计

### 5.1 shared basis 拆分

当前 shared basis bank 是一个统一集合。第一版改为逻辑上拆成两组：

- `B_side`
- `B_free`

推荐的第一版最小配置：

- `side_basis_count = 2` 或 `4`
- `free_basis_count = total_basis_num - side_basis_count`

不要求两组的 quantizer 彻底分离。第一版更稳妥的做法是：

- 继续共用一个 shared latent / residual FSQ
- 在 basis mixing 头层面分出 side path 与 free path

这样可以尽量复用现有 `DistNet` 主体，避免改动过大。

第一版需要把 “`B_side` 到底是哪一组 basis” 写成可实现规则。推荐采用：

- `B_side` 由一个单独的 `side basis bank` 显式持有，大小固定为 `2` 或 `4`
- `B_free` 继续使用当前 `levels=(2,3,6)` 的 level-structured basis bank
- 第一版不采用“从现有 level-wise basis bank 中切若干 basis 作为 side basis”的做法

这样可以避免额外回答“`side_basis_count` 在各个 level 如何分配”的问题，也能保持当前 residual FSQ 与 level-wise decode 主干不变。

### 5.2 reconstruction 路径

shared reconstruction 改成两段相加：

- `shared_side_reconstruction`
- `shared_free_reconstruction`

最终：

- `shared_reconstruction = shared_side_reconstruction + shared_free_reconstruction`
- `reconstructed = shared_reconstruction + private_weight * private_residual`

这样后续可以直接分析：

- side 语义究竟由哪部分 basis 承担
- free shared 分支是否仍在偷学 side

第一版建议 side 路径的形式尽量简单：

- `shared_quantized -> side mixing head -> side coefficients / weights -> side basis bank`

不要求 side path 与 free path 共享完全相同的 basis 选择机制。第一版的优先级是“先让 side path 显式存在并可分析”，而不是“完全复用现有 free path 的 level-wise routing”。

### 5.3 dataset 路径

dataset disentanglement 是本设计的中期目标，但**不进入第一版最小闭环**。

原因有两点：

1. 当前 `v19` 系列实验并未默认启用 dataset auxiliary
2. 若把 `B_side/B_free`、group-level side supervision、dataset auxiliary re-enabled 同时放进第一版，将导致实验归因过于混杂

因此第一版采取更保守的策略：

- 不重新启用 `private_dataset_loss`
- 不重新启用 `shared_dataset_adv_loss`
- 先通过后验 probe 观察 `B_side`、`B_free`、`private_z` 各自的 dataset 可分性

如果第一版能确认：

- `B_side` 确实承载稳定侧别语义
- `B_free` 没有明显完全接管 side

再在第二版加入：

- `private -> dataset classifier`
- `shared grouped rep -> dataset adversary`

### 5.4 side 输出头

第一版 side 分类不再直接从每帧 `shared_quantized` 读出，而是从 side-specific 的**group-level 聚合表示**读出。

推荐流程：

1. 得到每帧 side-specific activation / logits / coeffs
2. 用 `valid_mask` 对时间维做 masked pooling
3. 得到 group-level side representation
4. 从该 group-level representation 预测 `side`

这个设计和当前训练样本粒度一致，也更接近“这个 group 所对应 subject 的 shared side basis 长什么样”。

为避免第一版目标被旧监督路径稀释，需要明确：

- 当前基于逐帧 `shared_quantized` 的 continuous side head 不再作为主 side 监督
- 当前基于 `decoded_indices` 的 discrete side head 不再作为主 side 监督

其中离散 side head 当前与 `levels[1]` 强绑定，不再适合作为第一版 debug 指标。第一版默认应：

- 停用旧的 discrete side 主头
- 不把它纳入默认日志指标

若后续需要 ablation，应显式说明它是在旧 level-wise 语义下的对照头，而不是 `B_side` 的代理。

---

## 6. 监督设计

### 6.1 保留的损失

以下损失保持存在：

- `recon loss`
- `shared reconstruction loss`
- `lq loss`
- `orth loss`
- `basis_l1 loss`
- `private residual regularization`

### 6.2 新的主监督目标

第一版新增或替换的主要监督是：

- `group-level side loss`

其原则是：

- 只允许 `B_side` 对应的表示去预测 `side`
- 不再要求 `B_free` 直接服务于 `side`
- 不再以逐帧 CE 为主
- 默认替换现有两路 frame-wise `side_cont / side_disc` 主监督

### 6.3 序列聚合方式

由于标签是 `subject-level` 常量、而训练样本是 `group_size=4` 的 grouped sequence，第一版统一使用 masked temporal pooling：

- 输入：`B x T x D`
- mask：`valid_mask`
- 输出：`B x D`

推荐默认使用 masked mean。

第一版不引入更复杂的 temporal attention，避免把“可解释 basis”和“时序模块”耦合在一起。

需要注意：实现上这不只是“换一个 loss”，而是要改变当前训练协议中监督的聚合位置。

当前代码是：

- 先把序列展平成 frame batch
- 逐帧得到监督输出
- 再用 `valid_mask` 做 masked mean

第一版需要显式引入：

- per-frame side-specific representation
- group-level pooled representation
- group-level side supervision

因此这属于一次受控的训练协议扩展，而不是单纯新增一个 head。

### 6.4 dataset 的第一版处理

第一版不把 dataset auxiliary 纳入主损失。

dataset 相关判断先通过后验 probe 完成，例如：

- `B_side` pooled rep 的 dataset probe
- `B_free` pooled rep 的 dataset probe
- `private_z` 的 dataset probe

如果后验结果显示：

- `B_side` 明显带有 dataset 偏差
- 或 `private_z` 反而不如 shared 更可分

再进入第二版，引入显式的 `private_dataset_loss` 与 `shared_dataset_adv_loss`。

### 6.5 severity 的处理

第一版不把 `severity` 放入主损失。

原因：

- 当前结构还没有独立的 `B_severity`
- 若和 `side` 一起强推，容易出现语义耦合
- 当前样本规模和验证帧数不足，不适合一次引入两个 subject-level 主监督目标

但需要保留后续接口：

- group-level shared rep probe
- side-specific rep probe
- free rep probe

用于离线比较哪一部分对 `severity` 更敏感。

### 6.6 blendshape 的处理

当前不建议将 blendshape 作为第一版主监督。

原因：

- 标注可能不够准确
- 某些 blendshape 未必能稳定反映到距离矩阵里
- 当前主目标是先把 `side` 语义固定到 shared semantic basis

blendshape 在第一版更适合作为后验分析：

- 检查 side basis 是否同时对应特定肌肉激活模式
- 检查 free basis 是否更接近动作细节而非临床标签

---

## 7. 训练策略

### 7.1 第一阶段策略

基于当前 `v19 backbone`，第一阶段只增加 side semantic bank 相关改动：

- 保留现有 `v19` 的 backbone 与重建相关超参作为起点
- 新增 `B_side + B_free`
- side supervision 改成 group-level
- dataset auxiliary 保持关闭

### 7.2 为什么先不接 severity

`side` 比 `severity` 更容易形成可命名 basis，原因是：

- `side` 有更直接的空间不对称性
- mouth/x 模式下更容易体现在 basis heatmap 上
- 现有码本分析已经显示侧别可分性比严重度更强

因此第一版的成功标准应收窄为：

- `B_side` 是否形成稳定的、可分析的临床语义 basis

### 7.3 成功标准

第一版不以总 loss 最低作为首要目标，而以以下指标作为主判断依据：

1. `B_side` 对 `side` 的 group-level 预测稳定提升
2. `shared_side_reconstruction` 占 shared reconstruction 的比例可观且稳定
3. `basis_bank_heatmap` 中 `B_side` 比 `B_free` 更容易被命名
4. `B_side` pooled rep 的 side probe 强于 `B_free`

第二层指标：

- 总重建不要出现不可接受退化
- higher-level code usage 不应明显恶化

第一版还必须新增可计算分析输出，否则上述成功标准无法落地。至少包括：

- `shared_side_reconstruction`
- `shared_free_reconstruction`
- `group_pooled_side_rep`
- `group_pooled_free_rep`
- `side_path_usage` 或 `side_path_weight_norm`
- `free_path_usage` 或 `free_path_weight_norm`

分析脚本需要能回答：

- side path 是否闲置
- free path 是否偷学 side
- side/free 两路各自承担了多少 shared reconstruction
- side/free 两路各自的 dataset 可分性如何

---

## 8. 风险与失败模式

### 8.1 风险 1：side 语义被 free branch 偷走

即使只让 `B_side` 接 `side loss`，`B_free` 仍可能在重建时携带大量 side 信息。

第一版先接受这个风险，但要通过后验分析检查：

- 用 `B_side` 和 `B_free` 各自的 group pooled rep 做 side probe

如果 `B_free` 仍然更强，第二版再考虑：

- `B_free` 去侧别约束
- `B_side/B_free` 去相关约束

### 8.2 风险 2：side path 直接闲置或塌缩

除了 free 偷学 side 外，另一个失败模式是：

- `B_side` 基本不参与 shared reconstruction
- side loss 只通过很弱的 shortcut 被满足

因此第一版必须显式监控：

- side path 的重建占比
- side path 的权重范数 / 激活占用
- side path 单独做 side probe 的效果

### 8.3 风险 3：side basis 只学到动作强度

由于动作模式单一，side basis 可能学到的是“幅度差异”而不是“左右不对称”。

这需要在后验分析中检查：

- 左右样本的 side basis heatmap 是否有镜像差异
- side pooled rep 是否主要沿左右分离，而不是只和幅度相关

### 8.4 风险 4：shared 中仍残留明显 dataset 信息

即使第一版不显式训练 dataset auxiliary，shared 的 side/free 两路仍可能携带 dataset 痕迹。

因此需要通过后验 probe 额外检查：

- `group_pooled_side_rep` 的 dataset 可分性
- `group_pooled_free_rep` 的 dataset 可分性
- basis heatmap 是否出现稳定的数据集风格偏移

### 8.5 风险 5：shared 能力下降后，private residual 再次接管

新增语义约束可能削弱 shared reconstruction 自由度，导致 private residual 回升。

因此第一版必须继续监控：

- `shared_recon`
- `scaled_private_residual`
- total recon

若 private residual 明显反弹，则说明 semantic bank 容量过小或 side loss 过强。

---

## 9. 实现边界

第一版预期主要改动模块：

- `scripts/lq/model/distnet.py`
- `scripts/lq/model/heads.py`
- `scripts/lq/training/losses.py`
- `scripts/lq/train.py`
- `scripts/lq/analyze_checkpoint.py`

可能需要新增：

- group pooled rep 的输出字段
- `shared_side_reconstruction` / `shared_free_reconstruction` 的分析输出
- side semantic bank 的配置参数

第一版尽量**不改**：

- dataset 数据接口
- residual FSQ 主体实现
- 现有 basis init 文件格式

---

## 10. 后续扩展

若第一版成功，第二版再考虑：

1. 新增 `B_severity`
2. 将 `severity` 改为 ordinal supervision
3. 给 `B_side` 加镜像配对约束
4. 对 `B_free` 加去侧别约束
5. 用 blendshape 做后验语义对齐或弱辅助监督

---

## 11. 推荐结论

推荐按以下顺序推进：

1. 以 `v19 backbone` 为起点
2. 新增 `B_side + B_free`
3. 将 `side` 监督改成 group-level，仅绑定 `B_side`
4. 用后验分析回答：
   - `B_side` 是否真的学到侧别语义
   - `B_free` 是否仍在偷学侧别
   - `B_side` 是否带有明显 dataset 偏差
5. 若第一版结果稳定，再进入第二版：
   - 重新引入 dataset auxiliary
   - 再讨论 `severity` 主监督

这是当前数据条件下，最接近“可命名 shared basis”目标、同时风险最低的第一版方案。
