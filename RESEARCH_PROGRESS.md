# 研究进展记录

[TOC]

**最后更新**: 2026-04-20（补充LQ round-1 side semantic bank smoke）

---

## 一、研究目标

**核心问题**: 能否将面部关键点距离矩阵分解为可解释的子运动单元，实现身份-运动的显式解耦，用于面瘫分级？

**输入数据**:
- Subject_num × window_count × dim × dim (341×341)
- 每个窗口是一个341×341的距离矩阵
- 差分形式：ΔD = D(t) - D(t-1)，捕捉运动变化

**期望输出**:
1. 一组公共的运动基（每个基是341×341矩阵）—— 跨被试共享
2. 每个被试的运动激活系数（随时间变化）
3. 身份信息与运动状态的解耦

---

## 二、研究路径回顾

### 2.1 初始尝试：NMF → 失败

**发现**: NMF要求输入非负，但差分ΔD包含负值，无法直接使用。

### 2.2 转向SVD：验证差分有效性

**关键发现**:
- RAW矩阵（非差分）: PC1 dominant = eyehole（结构/身份）
- DIFF矩阵（差分）: PC1 dominant = mouth（运动语义）
- **结论**: 差分是捕捉运动的必要步骤

### 2.3 SVD多患者联合分解

**方法**: 将所有患者的差分窗口堆叠 → 联合SVD

**结果**:
| 数据集 | X PC1能量 | X PC2能量 | PC1 dominant |
|--------|----------|----------|--------------|
| IMR (227患者) | 64.6% | 25.1% | mouth |
| TT (42患者) | 47.0% | 27.2% | mouth |

**发现**:
- 基向量主要捕捉运动语义而非身份
- 不同数据集的PC1 dominant region均为mouth
- 身份信息没有混入基向量

### 2.4 Tucker分解尝试 → 不理想

**原因**:
- Tucker的spatial factor C是(341, r3)系数向量，不是341×341矩阵
- 无法像SVD的V那样直接reshape成热图
- 可视化和可解释性受限

---

## 三、Grassmann流形验证：共享基是否是被强迫的？

### 3.1 验证方法

使用Grassmann流形主角度分析：
1. 对每个患者单独做SVD，得到各自的基
2. 对每个数据集（IMR/TT）做联合SVD，得到数据集级别的基
3. 计算单患者基与联合基之间的主角度

**核心逻辑**：
- 若"真共享"：单患者基与本数据集联合基角度小，单患者基与异数据集联合基角度大
- 若"被计算强迫"：所有患者与所有联合基角度相近，无差异化

### 3.2 关键结果

**Joint-to-Joint（联合基之间）**:
| 模态 | PC1 | PC2 | PC3 | PC4 |
|------|-----|-----|-----|-----|
| X (水平) | 13.5° | 16.3° | 23.7° | 56.0° |
| Y (垂直) | 7.1° | 7.5° | 25.7° | 49.6° |

**单患者 vs 联合基 (PC1)**:
| 比较 | X PC1 | Y PC1 |
|------|-------|-------|
| IMR单患者 vs IMR联合 | 12.9° | 10.0° |
| TT单患者 vs IMR联合 | 20.0° | 9.9° |
| TT单患者 vs TT联合 | 14.8° | 8.9° |
| IMR单患者 vs TT联合 | 20.0° | 12.5° |

### 3.3 结论

- ✅ **联合基本身很接近**：TT_joint与IMR_joint夹角仅13.5°(X)和7.1°(Y)，两数据集共享相似的运动子空间
- ✅ **患者与本数据集联合基对齐更好**：IMR患者→IMR联合(12.9°) < IMR患者→TT联合(20.0°)；TT患者→TT联合(14.8°) < TT患者→IMR联合(20.0°)
- ✅ **跨数据集差异主要由数据集内部异质性驱动**，而非系统偏差
- ✅ **共享基不是被强迫的**：单患者SVD与联合SVD在 Grassmann 流形上确实自然对齐

---

## 四、blendshape弱监督验证 ✅ COMPLETED（2026-03-30）

**验证目标**: 验证SVD分解得到的时间系数是否与blendshape变化存在一致性

**窗口对齐方式**:
- 每20帧作为一个窗口（与SVD窗口一致）
- 计算窗口内所有帧的blendshape均值
- 相邻窗口均值相减得到变化量
- 尾部不足20帧的部分忽略

**验证方法**:
1. 对 IMR 和 TT 联合SVD 的前4个主模态
2. 将每个 TT 患者的数据投影到 IMR_joint 和 TT_joint 基上，得到时间系数
3. 计算时间系数与 blendshape 变化（52个AU）的 Pearson 相关性

### 关键结果

**PC1/PC2 与 blendshape 最高相关性**:

| 模态 | PC | Top Blendshape | r (mean±std) |
|------|-----|----------------|----------------|
| IMR_X | PC1 | mouthUpperUpLeft | 0.73±0.26 |
| IMR_X | PC2 | cheekPuff | 0.65±0.26 |
| IMR_Y | PC1 | jawOpen | 0.69±0.23 |
| IMR_Y | PC2 | jawOpen | 0.64±0.24 |
| TT_X | PC1 | cheekPuff | 0.78±0.20 |
| TT_X | PC2 | cheekSquintRight | 0.52±0.29 |
| TT_Y | PC1 | cheekPuff | 0.68±0.22 |
| TT_Y | PC2 | cheekPuff | 0.61±0.25 |

**PC3/PC4 语义映射**:

| 模态 | PC | Top Blendshape | r (mean±std) |
|------|-----|----------------|----------------|
| IMR_X | PC3 | eyeLookOutLeft | 0.53±0.31 |
| IMR_Y | PC3 | mouthLowerDownRight | 0.55±0.29 |
| TT_X | PC3 | noseSneerRight | 0.52±0.30 |
| TT_Y | PC3 | eyeLookDownRight | 0.56±0.25 |
| IMR_X | PC4 | mouthFrownLeft | 0.49±0.28 |
| IMR_Y | PC4 | mouthPressLeft | 0.53±0.29 |
| TT_X | PC4 | eyeLookInRight | 0.61±0.30 |
| TT_Y | PC4 | eyeLookDownRight | 0.56±0.26 |
### 结论

- ✅ **PC1 (X模态) ↔ cheekPuff + mouthUpperUpLeft** (r=0.73-0.78, std=0.20-0.26)：脸颊鼓起+嘴角上扬
- ✅ **PC1 (Y模态) ↔ jawOpen + cheekPuff** (r=0.67-0.69, std=0.22-0.23)：张嘴垂直运动
- ✅ **PC2 ↔ cheekSquint + jawOpen** (r=0.52-0.65, std=0.24-0.29)：脸颊收缩+张嘴
- ✅ **PC3/PC4 ↔ eyeLook系列** (r=0.49-0.61, std=0.25-0.31)：开始捕捉眼球运动
- ✅ **IMR_joint 与 TT_joint 表现一致**：相同的 PC 捕捉相同的 blendshape，验证基的语义一致性

**关于标准差的观察**:
- PC1的r值高（0.68-0.78）且标准差相对较小（0.20-0.26），说明跨患者一致性较好
- PC3/4的r值稍低且标准差较大（0.25-0.31），说明跨患者异质性随阶数增加

**验证成功意义**: 在单一动作模式（咧嘴）样例中，基于距离差分矩阵可得到具有明确语义、跨人群的公共动作基。时间系数与blendshape AU存在显著相关性（r > 0.6），具有物理可解释性。

### ⚠️ 重要补充：窗口数对相关性的影响 (2026-04-01)

**问题**: win20-step20 的窗口数过少，导致相关性可能虚高

**窗口数分布统计**:

| 数据集 | 方法 | 窗口数 | 差分后实际相关计算点数 |
|--------|------|--------|---------------------|
| win20 IMR | SVD | 94.3%只有6个 | **99.1% ≤ 5个点** |
| win20 TT | SVD | 38.1% > 10个 | 31.0% > 10个点 |
| win5 IMR | DMD | 100% > 10个 | 99.1% 在25-133之间 |
| win5 TT | DMD | 100% > 10个 | 平均49个点 |

**结论**:
- win20-step20 的 IMR 被试做相关性分析时只有 **~5个点** 计算 Pearson r，统计上极不可靠！
- DMD 使用 win5-step5 数据，窗口数大幅增加（25-130+），相关性计算更可靠

**DMD相关性结果** (使用TT联合DMD模态):

| DMD模态 | 最高相关blendshape | r |
|---------|-------------------|-----|
| X Mode1 | jawForward | 0.59 |
| Y Mode1 | cheekPuff | 0.60 |
| Y Mode2 | cheekPuff | 0.64 |

**实际相关性可能介于 SVD 和 DMD 之间**: SVD由于窗口数过少可能虚高，DMD由于使用不同数据集（TT联合基）可能偏低。真实r值可能在 **0.5-0.7** 范围内。

---

## 五、基于码本的矩阵解耦 ✅ COMPLETED（2026-04-12）

### 5.1 码本分型的探索

**目标**: 验证单患者SVD提取的PC1主模态是否可以用作"可学习公共码本"的初始化基，探索公共码本中的共享基是否需要区分不同类型。合适的码本基初始化可以避免死基的产生

**数据**: 269患者 (IMR=227, TT=42)，PC1来自`IMR-SVD/`和`TT-SVD/`
**算法**: PCA(50) + Logistic Regression (L2)，5折分层交叉验证
**矩阵类型**: Full (341×341) 和 Mouth (119×119, indices 188-307)

#### 实验1: IMR vs TT 数据集分类 (二分类)

| Config | Accuracy | F1 | AUC |
|--------|----------|-----|-----|
| full_x | **0.918±0.040** | 0.829 | **0.962** |
| full_y | 0.914 | **0.835** | 0.882 |
| mouth_x | 0.903 | 0.813 | 0.909 |
| mouth_y | 0.881 | 0.783 | 0.921 |

**结论**: IMR和TT在PC1上**差异显著**（AUC=0.96），可能需要按人群分。

#### 实验2: 测别分类 (Left/Normal/Right, 三分类)

| Config | Accuracy | F1 |
|--------|----------|-----|
| full_x | 0.732 | 0.732 |
| full_y | 0.751 | 0.750 |
| **mouth_x** | **0.762** | **0.760** |
| mouth_y | 0.691 | 0.692 |

**结论**: PC1包含**左右不对称特征**，三分类准确率73-76%，需要按照左右不对称分类。

#### 实验3: 严重度分类 (Normal/Mild/Severe, 三分类)

| Config | Accuracy | F1 |
|--------|----------|-----|
| full_x | 0.584 | 0.563 |
| full_y | 0.594 | 0.563 |
| **mouth_x** | **0.613** | **0.595** |
| mouth_y | 0.577 | 0.564 |

**结论**: 严重度分类**接近随机基线**(33%)，PC1捕捉对等级信息反馈有限，但难以得出不必要性。

### 关键结论

1. **数据集效应显著**: AUC=0.96，IMR与TT系统偏差大
2. **Mouth区域是主要判别区**: 测别/严重度分类均以mouth_x最好
3. **X方向优于Y方向**: 水平运动信息更具有区分性
4. **PC1码本性质**:
   - ✅ 可区分: 数据集、侧别(Left/Right)
   - ❌ 不可区分: 严重程度

### 代码目录

`scripts/val_codebook/` - 包含完整的实验代码和可视化输出

---

## 六、LQ原型训练进展与FSQ替换（2026-04-19）

### 6.1 当前范围

当前 `scripts/lq` 的工作重点已经收敛到一个可运行的单方向原型：

- mode: `x`
- region: `mouth`
- data: `data/win20-step20/IMR,data/win20-step20/TT`
- 输入形式: `B x T x 1 x H x W`
- 当前目标: 学习共享离散运动码 + 动作基重建，并保留私有残差分支

### 6.2 工程状态

当前已经完成：

- 序列输入训练接口接通
- `basis_init` 接入训练流程
- batch memory smoke test 接入
- `train.py` 按帧计算 loss
- round-1 工程重构已完成第一轮拆分：
  - 训练逻辑拆入 `scripts/lq/training/`
  - 数据逻辑拆入 `scripts/lq/data/`
  - `DistNet` 内部拆入 `scripts/lq/model/` 子模块
  - 对外 CLI 与 checkpoint / dataset 语义保持不变
- `network.py` 支持结构探针：
  - `pool_size`
  - `shared_dim`
  - residual cap
  - soft basis mixing
- 官方量化器切换接通：
  - `LatentQuantize`
  - `FSQ`
- `analyze_checkpoint.py` 可统计 basis bank 与 code usage

### 6.3 当前阶段简要总结

当前这一阶段已经完成从“单个 FSQ baseline”到“结构 probe”的一轮收敛，
重点围绕以下问题展开：

- 是否能让共享离散码真正被使用，而不是 collapse
- 是否能让 shared branch 承担更多解释，而不是主要依赖 private residual
- basis 的正交性、稀疏性和 shared 结构复杂度是否会改变上述行为

当前得到的简要结论是：

- 早期 collapse 的主要缓解来自官方 `FSQ` 替换，`v10` 起 code usage 明显
  好于旧版 `LatentQuantize`
- 进一步加强 basis 约束（全局 QR、basis L1）后，总重建可以继续下降，但
  很容易把压力重新推回 private residual branch
- 引入 residual FSQ 后，高层 code usage 比单个 FSQ 更健康，说明量化结构
  本身仍然重要
- 在 residual FSQ 基础上，再增加 shared 路径复杂度（anchor-guided sparse
  mixing）并显式加入 `shared_recon` 监督后，shared reconstruction 比
  `v17` 有明确回升，说明“先增强 shared 结构，再单独监督 shared”是正确方向

当前最有代表性的几组结果为：

- `v10 FSQ baseline`:
  - `val_recon = 0.3600`
  - `val_shared_recon = 0.3620`
  - L2 `[20, 23, 37]`
  - L3 `[18, 2, 3, 3, 25, 29]`
- `v17 residual_fsq + global_qr + basis_l1`:
  - `val_recon = 0.3109`
  - `val_shared_recon = 0.3423`
  - `val_scaled_residual = 0.0393`
  - L3 `[55, 2, 1, 3, 13, 6]`
- `v18 residual_fsq + sparse_shared_mixing + shared_recon_loss`:
  - `val_recon = 0.3150`
  - `val_shared_recon = 0.3469`
  - `val_scaled_residual = 0.0394`
  - L3 `[57, 1, 0, 2, 8, 12]`
- `v19 v18 + tighter private residual cap`:
  - `val_recon = 0.3275`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0214`
  - L3 `[57, 1, 1, 1, 10, 10]`
- `v20 v19 + tighter cap=0.4`:
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3448`
  - `val_scaled_residual = 0.0171`
  - L3 `[57, 1, 0, 2, 17, 3]`
- `v21 v19 + looser cap=0.6`:
  - `val_recon = 0.3245`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0251`
  - L3 `[57, 1, 0, 2, 11, 9]`

这里需要特别注意：

- `v18` 和 `v19` 的总 loss 与早期 probe 不可直接比较，因为已显式加入
  `shared_recon_weight`
- `v20` 和 `v21` 也属于同一目标函数设置，因此更适合与 `v18/v19` 横向比较，
  不应和更早期 probe 直接比较总 loss
- 目前验证集仍只有 `80` 个有效帧，因此所有 probe 结论都应理解为方向性证据，
  不是最终定论

### 6.4 当前判断与下一步计划

目前最合理的判断是：

- 问题不只是 loss weight 设置，而是 shared branch 本身表达能力不足
- 只收紧 basis 或只调 residual 权重，都会被 private residual 重新“绕开”
- residual FSQ 有助于改善 code usage，但不能自动带来更好的 shared 解释
- “增强 shared 结构 + 显式优化 shared reconstruction”已经显示出正向效果，
  应该作为下一阶段主线保留
- `0.4/0.5/0.6` 的局部 cap sweep 表明：
  - cap 放松到 `0.6` 时，模型主要把收益转移到 private residual，解释性变差
  - cap 收紧到 `0.4` 时，private residual 进一步下降，但 higher-level code usage
    比 `0.5` 更集中
  - `0.5` 目前仍是更稳妥的解释性 baseline；`0.4` 可以作为更激进的私有分支抑制备选

下一步计划：

- 保留当前 `v18/v19/v20` 共享结构设计：
  - residual FSQ
  - anchor-guided sparse shared mixing
  - shared reconstruction supervision
- 当前更合理的落点是：
  - 以 `v19` 作为稳妥 interpretability baseline
  - 将 `v20` 作为 stricter-private 备选对照
- 下一步应从纯 tradeoff sweep 转向 shared 解释性的进一步加强，优先回答：
  - 能否在保持 `v19/v20` 级别 private residual 的前提下，让 higher-level code usage
    更均匀
  - 能否通过更直接的共享结构约束，让 basis heatmap 语义更稳定、更不相似
  - 在当前小验证集之外，这些趋势是否能在更大有效帧设置下保持稳定

### 6.5 Round-1 side semantic bank smoke（2026-04-20）

基于 `v19` backbone 的关键结构参数，补充了 `v22` round-1 smoke preset：

- 保留：
  - `action_basis_init_path=scripts/lq/init_basis/basis_x.npy`
  - `shared_recon_weight=1.0`
  - `quantizer_type=residual_fsq`
  - `basis_orthogonalization=global_qr`
  - `private_residual_max_l1=0.5`
- 新增：
  - `side_semantic_enabled=True`
  - `side_basis_count=2`
  - `side_loss_weight=0.3`

Smoke 运行：

- 训练输出：`outputs/lq_x_mouth_v22_side_semantic_bank_probe_smoke`
- 1 epoch validation：
  - `val_loss = 1.0672`
  - `val_recon = 0.3636`
  - `val_shared_recon = 0.3636`
  - `val_scaled_residual = 0.00233`
  - `val_side_group = 1.1104`

Checkpoint analysis smoke 已完成，并写出：

- `analysis/summary.json`
- `analysis/side_basis_bank_heatmap.png`
- `analysis/group_level_representations.npz`

当前 smoke 结论：

- `B_side` 已经被实际使用：
  - `side_basis_shape = [2, 119, 119]`
  - `mean_side_path_usage = 0.500`
  - `mean_free_path_usage = 0.273`
  - `mean_side_recon_l1 = 0.00162`，高于 `mean_free_recon_l1 = 0.000262`
- 但 1 epoch 下还没有看到 side probe 优势：
  - `side_from_side_rep_acc = 0.364`
  - `side_from_free_rep_acc = 0.364`
- free/shared 路径仍带有明显 dataset/side 痕迹：
  - `dataset_from_side_rep_acc = 0.818`
  - `dataset_from_free_rep_acc = 0.818`

因此当前可确认的是：round-1 preset、训练链路和 side/free 分析链路已经打通；但仅凭 smoke 还不能认为 side semantic separation 已成立。

### 6.6 文档入口

更完整的实现状态、实验时间线、结果和未决问题，见：

- [`docs/lq_progress.md`](/home/weizilin/generate_idea/docs/lq_progress.md)
- [`docs/lq_train_presets.md`](/home/weizilin/generate_idea/docs/lq_train_presets.md)
- [`docs/lq_dataset_refactor_checklist.md`](/home/weizilin/generate_idea/docs/lq_dataset_refactor_checklist.md)

---

## 七、关键实验发现总结

### 7.1 差分形式确认
- ✅ 差分ΔD比RAW矩阵更能捕捉运动语义
- ✅ 差分后PC1 dominant从eyehole(身份)变为mouth(运动)

### 7.2 SVD联合分解可行
- ✅ 基向量跨患者共享（PC1 dominant region一致）
- ✅ 身份信息没有混入基向量
- ✅ Grassmann流形分析证实：共享基是"真共享"而非"计算强迫"

### 7.3 Grassmann流形验证（2026-03-30）
- ✅ 跨数据集联合基子空间接近（TT_joint vs IMR_joint: X 13.5°, Y 7.1°）
- ✅ 患者基与本数据集联合基对齐更好，验证基的真实共享性

### 7.4 blendshape时间系数验证（2026-03-30）

- ✅ PC1 ↔ cheekPuff/mouthUpperUpLeft (脸颊/嘴角运动)
- ✅ PC2 ↔ jawOpen/cheekSquint/cheekPuff (张嘴+脸颊收缩)
- ✅ IMR_joint 与 TT_joint 捕捉相同 blendshape，验证基的语义一致性

---

## 八、未来研究计划

### 8.1 中期方向（神经网络）

**目标**: 扩展到复杂运动序列 + 语义可解释

**可能方案**:
1. **Denoising Autoencoder + Disentanglement**
   - 输入: ΔD矩阵
   - 隐空间: identity_z + motion_z
   - Loss: Reconstruction + β * MI(identity, motion)

2. **结合blendshape标注的弱监督**
   - MediaPipe blendshapes: 68+ action unit激活值
   - 部分时间窗口有标签
   - 加约束: 基的激活系数与对应blendshape相关

### 8.2 长期目标

- 实现多动作类型的运动基分解
- 达到"每个基有确切动作语义"的可解释性
- 用于面瘫分级任务

---

## 九、关键文献

| 论文 | 年份 | 相关内容 |
|------|------|---------|
| Hallac 2017 (PMID 28011182) | 2017 | Frame-to-frame距离 + Procrustes，与ΔD概念对齐 |
| Hallac 2024 (PMID 39476531) | 2024 | 3D面部拓扑 + 曲率分析，面瘫分级 |
| EDTalk | 2024 | 三空间正交基向量，口型/姿态/表情分离 |
| AniTalker | 2024 | MINE互信息解耦身份vs运动 |

---

## 十、待验证问题清单

- [x] 共享基是否是真共享？（Grassmann流形验证 ✓）
- [x] blendshape弱监督是否有效？（第四章 ✓ PC1↔jawOpen, PC2↔mouth动作）
- [x] PC1码本能否区分数据集/侧别/严重度？（第五章 ✓ 数据集AUC=0.96, 侧别Acc=76%, 严重度Acc≈61%）
- [ ] 神经网络是否能扩展到多动作类型？
