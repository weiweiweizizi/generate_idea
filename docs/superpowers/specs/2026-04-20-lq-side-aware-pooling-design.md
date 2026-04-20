# LQ Side-Aware Pooling Design

**状态**: 草稿

**日期**: 2026-04-20

---

## 1. 背景

当前有效 baseline 仍是 `v26 early-branch`：

- `data = data/win20-step20/IMR,data/win20-step20/TT`
- `mode = x`
- `region = mouth`
- `basis_size = 119`
- `quantizer_type = residual_fsq`
- `levels = (2, 3, 6)`

在这条线上：

- `free` 分支已经通过 `residual FSQ + shared basis` 学到了比较稳定的 shared motion 路径
- `side` 分支已经从旧的 late-split 结构中分离出来
- 但 `dataset_from_side_rep` 仍然偏高，说明 side 路径仍在携带 dataset 信息

上一轮 `v27 side_z_dim=8` 的结果说明：

- 单纯压缩 `side_z` 容量并不能压低 dataset leakage
- `side_from_side_rep` 下降
- `dataset_from_side_rep` 没降
- `val_recon` 和 `val_side_group` 也明显变差

因此问题很可能不在“side latent 太大”，而在两个更早的环节：

1. side 相关空间结构在 shared trunk 中被过早平均 / 平滑
2. 当前 `side_basis_logits` 只有 2 维，side path 表达过于狭窄，不利于展开更细的 side 语义

---

## 2. 核心判断

这轮设计基于一个很明确的判断：

- 不改 shared trunk 的 `side-aware pooling` 不能凭空恢复已经丢失的信息
- 但它可以回答一个关键问题：
  - `layer3` 的 `feats [B*T, 32, 15, 15]` 中，是否仍保留了足够的 side 空间结构，只是当前 `2x2 pooling` 读得太粗

因此，这一轮不把它当作最终解，而把它当作一个**结构 probe**：

- 如果 side-aware pooling 明显改善 side probe / side group，而不严重损伤重建
  - 说明 trunk 中仍有可读的 side 信息
  - 下一步可以继续沿着 “更强 side readout / 更强 side branch” 推进
- 如果 side-aware pooling 基本无效
  - 说明问题更可能出在 trunk 已经过早抹平了 side 信息
  - 下一步应转向更早分支或显式左右差分结构

---

## 3. 目标

本轮目标不是直接追求最优指标，而是验证：

> 在不改 shared trunk 的前提下，使用更尊重已知 mouth 左右空间排布的 side readout，并把过窄的 side basis 容量从 `2` 扩到 `4` 后，是否能改善 side 语义。

因此这轮实验必须被定义为：

- 一个**联合结构 probe**
- 修改项同时包含：
  - `side-aware pooling`
  - `side_basis_count = 2 -> 4`

也就是说，这不是“纯 pooling 单变量实验”。

如果这轮有效，结论只能写成：

- `side readout + side basis capacity` 的联合修改有效

不能把收益直接全部归因于 pooling。

第一版只做两类改动：

1. 把当前 `side_pool(2x2)` 替换成固定区域的 side-aware pooling
2. 把 `side_basis_count` 从 `2` 提升到 `4`

同时保持：

- `free` 分支不改
- `residual FSQ` 不改
- `private` 分支不改
- shared trunk 不改
- 当前 canonical side rep 分析口径不改

---

## 4. 现有问题点

当前 `early-branch` 的 side 路径是：

- `feats -> side_adapter -> side_pool(2x2) -> flatten -> side_head -> side_latent`
- `side_latent -> side_basis_logits(2) -> side_path_usage(2) -> side_path_representation(2)`

这里存在两个连续瓶颈：

1. `feats [32, 15, 15] -> AdaptiveAvgPool2d(2, 2)`
   - 把较粗的空间结构再次压缩
   - 不尊重 mouth crop 内部的左右语义块
2. `side_basis_count = 2`
   - 最终 side path 只有 2 维 usage
   - 表达过于接近 0/1 laterality knob
   - 不利于展开更细粒度的 side 结构

---

## 5. 输入空间先验

当前 `mouth` 区域来自全脸矩阵的 `[188:307, 188:307]`，局部大小为 `119 x 119`。

用户给出的 mouth 内部空间先验是：

- `188:210`：嘴部周围左边
- `210:233`：嘴部周围右边
- `233:270`：嘴部左边
- `270:307`：嘴部右边

映射到当前 mouth-local 索引后，对应为：

- `around_left = [0:22)`
- `around_right = [22:45)`
- `mouth_left = [45:82)`
- `mouth_right = [82:119)`

这轮设计要求 side 分支显式尊重这四块结构。

---

## 6. 主方案

### 6.1 总体思路

保持 shared trunk 不动，只修改 side branch 的 readout：

- 输入仍然是 `feats [B*T, 32, 15, 15]`
- side branch 不再使用 `AdaptiveAvgPool2d((2, 2))`
- 改为 4 个固定区域 token pooling
- 4 个 token 拼接后再送入 `side_head`

同时把：

- `side_basis_count = 2 -> 4`
- `side_z_dim` 恢复为 `32`

这里明确不继续使用 `side_z_dim=8`，因为这一轮要测的是 pooling/readout，而不是 side latent 容量。

### 6.2 4-token fixed region pooling

#### 6.2.1 feature map 尺度

当前 trunk 输出：

- `input [B*T, 1, 119, 119]`
- `initial_conv(stride=2) -> [B*T, 8, 60, 60]`
- `layer1(stride=2) -> [B*T, 16, 30, 30]`
- `layer2(stride=2) -> [B*T, 32, 15, 15]`
- `layer3(stride=1) -> [B*T, 32, 15, 15]`

side-aware pooling 的输入固定为：

- `side_feats = side_adapter(feats)`, shape `[B*T, 32, 15, 15]`

#### 6.2.2 block 定义

在第一版里，不做 learnable attention，也不做软 mask。

直接在 `15 x 15` feature map 上定义 4 个**固定、不重叠、可解释**的对角 block：

- `token_1`: `around_left × around_left`
- `token_2`: `around_right × around_right`
- `token_3`: `mouth_left × mouth_left`
- `token_4`: `mouth_right × mouth_right`

第一版采用近似整数映射：

- `around_left -> [0:3)`
- `around_right -> [3:6)`
- `mouth_left -> [6:10)`
- `mouth_right -> [10:15)`

因此 4 个 block 分别是：

- `[0:3, 0:3]`
- `[3:6, 3:6]`
- `[6:10, 6:10]`
- `[10:15, 10:15]`

#### 6.2.3 pooling 方式

每个 block 使用固定 masked average pooling：

- 输入：`side_feats [B*T, 32, 15, 15]`
- 对每个 block 在空间维 `(H, W)` 上求平均
- 输出一个 token：`[B*T, 32]`

最终得到：

- 4 个 token，每个都是 `[B*T, 32]`
- concat 后变成 `[B*T, 128]`

再接 side head：

- `side_head(128 -> 32 -> side_z_dim)`
- `side_z_dim = 32`

### 6.3 Side path 输出

这轮把 `side_basis_count` 提升到 `4`，因此：

- `side_basis_bank`: `[4, 119, 119]`
- `side_basis_logits`: `[B*T, 4]`
- `side_path_usage`: `[B*T, 4]`
- `side_path_representation`: `[B*T, 4]`

group-level canonical side rep 仍按当前口径：

- `masked_mean_per_sequence(side_path_representation, valid_mask)`

---

## 7. 为什么第一版不用差分 token

这轮明确**先不用**手工 laterality difference token，例如：

- `mouth_left - mouth_right`
- `around_left - around_right`

原因是这一轮的目标很明确：

- 先判断 `15x15` feature map 中的 side 信息还能否被更合理的 pooling 读出来

如果第一版一开始就加入显式差分 token，那么结果会混入更强的人为归纳偏置，难以判断：

- 到底是 side-aware pooling 有效
- 还是手工左右差分本身在起作用

因此第一版必须保持 probe 尽量干净。

---

## 8. 后备方案

如果主方案效果不明显，则进入第二版后备方案：

- 保留 4-token fixed region pooling
- 在此基础上显式构造 laterality contrast token：
  - `mouth_left - mouth_right`
  - `around_left - around_right`

也就是说，第二版会把 side branch 的输入从“局部块 token”推进到“局部块 + 左右差分 token”。

这一步的含义是：

- 如果单纯结构化读取不够
- 则显式把 laterality inductive bias 注入 side branch

但这一步不属于第一版主实验。

进入后备方案的触发条件写死为：

- 第一版 50 epoch best checkpoint 相对固定 `v26 reanalyzed` 基线，没有带来可接受的 side probe 改善
- 或 `dataset_from_side_rep_acc` 没有下降
- 或 `val_recon` / `val_side_group` 超出本 spec 第 10 节定义的容忍阈值

---

## 9. 不包含内容

这一轮明确不包含：

- 更早的 side branch 分流
- 修改 shared trunk 结构
- free 分支结构变动
- residual FSQ 替换或重构
- dataset adversarial 新设计
- side_z_dim 的继续压缩
- 复杂 attention pooling

原因是这一轮必须单独回答：

> 不改 trunk，仅通过更好的 side readout，以及略微放宽 side basis 表达容量后，是否还有救。

---

## 10. 比较基线与成功标准

这一轮的成功标准是相对固定的 `v26 reanalyzed` 基线，而不是绝对最优。

固定比较对象：

- checkpoint:
  - `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/best.pt`
- analysis:
  - `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/summary.json`

固定基线数值：

- `side_from_side_rep_acc = 0.8227`
- `side_from_free_rep_acc = 0.3273`
- `dataset_from_side_rep_acc = 0.8182`
- `dataset_from_free_rep_acc = 0.8091`
- `dataset_from_private_rep_acc = 0.8909`
- `val_loss = 0.7623`
- `val_recon = 0.2873`
- `val_side_group = 0.5365`

优先观察：

1. `side_from_side_rep_acc` 是否上升
2. `side_from_side_rep_acc > side_from_free_rep_acc` 是否继续成立
3. `dataset_from_side_rep_acc` 是否下降
4. `val_recon` 是否没有明显恶化
5. `val_side_group` 是否没有明显恶化

这里不要求一次性解决全部 leakage。

为了避免“没有明显恶化”过于模糊，这轮阈值固定为：

- `val_recon(candidate) - val_recon(v26_reanalyzed) <= 0.01`
- `val_side_group(candidate) - val_side_group(v26_reanalyzed) <= 0.10`

只要能证明：

- `side-aware pooling + side_basis_count=4` 让 side canonical rep 更可读
- 且不会显著损伤重建主线

就足以支持下一步继续在 side readout / side branch 方向推进。

---

## 11. 风险

主要风险有三类：

1. `15x15` feature map 中其实已经没有足够的 side 空间结构
   - 那么 fixed region pooling 不会带来改善
2. `side_basis_count = 4` 带来更大表达能力，但主要增加的是 leakage 而不是 side 语义
3. 固定 block 的空间映射过于粗糙
   - 可能导致 side readout 不稳定

如果第一版无明显收益，结论不应是“side 信息不存在”，而应更准确地写成：

- 在当前 trunk 输出尺度与当前固定 block 设计下，`side-aware pooling + side_basis_count=4` 的联合修改不足以挽救 side 信息

这时才进入第二版 laterality contrast 方案，或者转向更早分支。

---

## 12. 实施范围

第一版预计只修改：

- `scripts/lq/model/distnet.py`
- 新增一个新的训练 preset 脚本
- `RESEARCH_PROGRESS.md`

新的候选 run 命名固定为：

- preset:
  - `scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh`
- output:
  - `outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50`

必要时可以把 side-aware pooling 的小工具拆到：

- `scripts/lq/model/encoder.py`
  或
- `scripts/lq/model/heads.py`

但不应进行大规模重构。
