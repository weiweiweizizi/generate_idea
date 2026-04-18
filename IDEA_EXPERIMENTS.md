# Idea Experiments Log

[TOC]

**Last Updated**: 2026-03-31

> 📌 **脚本与结果目录映射**
> | 实验 | 脚本 | 结果目录 |
> |------|------|---------|
> | Idea 1 单患者验证 | `scripts/svd_single_patient.py` | `data/win20-step20/svd_single_patient_results/` |
> | Idea 1 多患者联合 | `scripts/svd_multi_patient.py` | `data/win20-step20/svd_multi_patient_results/` |
> | Idea 3 RAW vs DIFF | `scripts/svd_single_patient_raw.py` | `data/win20-step20/svd_single_patient_raw_results/` |
> | Idea 4 Grassmann | `scripts/grassmann_cross_analysis.py` | `data/win20-step20/grassmann_cross_analysis_results/` |
> | 早期Grassmann | `scripts/grassmann_analysis.py` | `data/win20-step20/grassmann_analysis_results/` |
> | blendshape相关性 | `scripts/blendshape_correlation_analysis.py` | `data/win20-step20/blendshape_correlation_results/` |
> | NMF基线(失败) | `scripts/nmf_baseline_x_y.py` | `data/win20-step20/nmf_results/` |
> | Tucker尝试 | `scripts/tucker_multi_patient.py` | `data/win20-step20/tucker_multi_patient_results/` |
> | Idea 1* DMD单患者 | `scripts/dmd_single_patient.py` | `data/win5-step5/dmd_single_patient_results/` |
> | Idea 1* DMD多患者 | `scripts/dmd_multi_patient.py` | `data/win5-step5/dmd_multi_patient_results/` |
> | Idea 1* DMD与blendshape相关性 | `scripts/dmd_blendshape_correlation.py` | `data/win5-step5/dmd_blendshape_correlation_results/` |

---

## Idea 1 验证：SVD差分基线（2026-03-28）

> **脚本**: `scripts/svd_single_patient.py`
> **结果目录**: `data/win20-step20/svd_single_patient_results/`

### 问题发现：NMF无法直接用于差分数据

**背景**：最初计划使用NMF分解关键点距离矩阵，但发现：
- NMF要求输入矩阵非负
- 差分数据 ΔD = D_t - D_{t-1} 包含负值
- 原始距离矩阵 D 虽然非负，但减去患者身份基准（均值）后同样产生负值

**尝试**：对原始距离矩阵D直接做NMF（Idea 1方向4）
- 结果：基向量语义混杂，所有10个患者的所有基都被所有样本激活
- 结论：原始D包含大量身份信息，不去除则基无法反映运动

### 转向SVD：差分+去身份基准

**思路转变**：
1. 用户之前在单患者上验证过SVD差分可发现语义（闭眼、动嘴）
2. SVD可以处理负值（无需非负约束）
3. 使用前后差分 ΔD 可以捕捉运动变化

**实验设计**：
- 数据：win20-step20，10个患者（9个IMR + 1个TT）
- 方法：对每个患者单独做SVD分解差分矩阵
- 分析：PC1-PC3能量占比、语义区域分布、时间系数

### 关键发现

#### 1. PC1能量高度集中

| 模态 | PC1平均能量占比 | 标准差 |
|------|----------------|--------|
| X（水平） | 96.1% | ±3.8% |
| Y（垂直） | 95.4% | ±4.7% |

TT/845380（20窗口）能量稍分散：X: 86%, Y: 83.7%，说明多窗口患者有更丰富的运动模式

#### 2. PC1语义集中于mouth区域

**所有10个患者的PC1 dominant region均为mouth**

| 患者 | X PC1 dominant | X dominant% | Y PC1 dominant | Y dominant% |
|------|----------------|-------------|----------------|-------------|
| IMR/00256 | mouth | 9.7% | mouth | 8.6% |
| IMR/00342 | mouth | 6.1% | mouth | 3.0% |
| IMR/00305 | mouth | 4.1% | mouth | 2.5% |
| IMR/00353 | mouth | 10.7% | mouth | 8.5% |
| IMR/00420 | mouth | 4.6% | mouth | 6.9% |
| IMR/00363 | mouth | 9.9% | mouth | 9.0% |
| IMR/00522 | mouth | 7.5% | mouth | 8.9% |
| IMR/00271 | mouth | 9.4% | mouth | 8.1% |
| IMR/00402 | mouth | 7.8% | mouth | 9.7% |
| TT/845380 | mouth | 10.9% | mouth | 9.9% |

**解读**：341维矩阵中mouth区域占74点（21.7%），PC1中mouth贡献约10%，与其他区域相比显著更高。这与"咧嘴动作"的语义一致。

#### 3. X vs Y模态差异

- **X模态（水平）**：PC1与outer_lips，嘴角横向运动
- **Y模态（垂直）**：PC1与mouth(inner+around)，垂直张嘴运动

### 当前结论

1. **SVD差分可发现语义**：PC1确实集中于mouth区域，说明"差分+SVD"可以捕捉运动语义
2. **单患者效果稳定**：10个患者结果一致，均显示mouth dominant
3. **PC1过于占优**（>95%）：可能意味着运动模式单一，或需要更多基才能分解更细的运动单元
4. **身份基准问题待解决**：当前方法依赖于单患者分析，多患者联合分解时身份是否会混入基中仍需验证

---

## Idea 1 补充：多患者联合SVD分解（2026-03-29）

> **脚本**: `scripts/svd_multi_patient.py`
> **结果目录**: `data/win20-step20/svd_multi_patient_results/`

### 实验设计

- 将IMR（227患者）和TT（42患者）分别做联合SVD分解
- X/Y模态分开处理
- 分析PC1-PC5的能量分布和语义区域

### 关键结果

#### IMR数据集 (227患者, 1164差分窗口)

| 模态 | PC1能量 | PC2能量 | PC3能量 | PC1 dominant |
|------|---------|---------|---------|--------------|
| X（水平） | 64.6% | 25.1% | 3.3% | mouth (9.4%) |
| Y（垂直） | 78.3% | 14.9% | 1.8% | mouth (8.9%) |

#### TT数据集 (42患者, 452差分窗口)

| 模态 | PC1能量 | PC2能量 | PC3能量 | PC1 dominant |
|------|---------|---------|---------|--------------|
| X（水平） | 47.0% | 27.2% | 8.3% | mouth (10.0%) |
| Y（垂直） | 69.1% | 18.4% | 4.3% | mouth (8.0%) |

#### 语义区域详细分布（前5基）

**IMR X模态:**
- PC1: mouth 9.4%, around_mouth 2.4%
- PC2: mouth 4.8%, around_mouth 1.1%
- PC3: eyehole 3.4%, mouth 2.6%
- PC4: mouth 4.8%, nose 1.7%
- PC5: mouth 2.2%, eyehole 1.6%

**TT X模态:**
- PC1: mouth 10.0%, around_mouth 2.3%
- PC2: mouth 2.2%, nose 0.8%
- PC3: eyehole 3.2%, mouth 2.3%
- PC4: mouth 2.2%, around_mouth 1.5%
- PC5: mouth 3.5%, eyehole 1.8%

**IMR Y模态:**
- PC1: mouth 8.9%, around_mouth 1.9%
- PC2: mouth 3.1%, nose 0.9%
- PC3: mouth 4.9%, around_mouth 1.6%
- PC4: mouth 4.2%, nose 1.1%
- PC5: mouth 6.5%, eyehole 1.7%

**TT Y模态:**
- PC1: mouth 8.0%, around_mouth 1.8%
- PC2: mouth 4.3%, nose 1.1%
- PC3: mouth 3.5%, around_mouth 1.7%
- PC4: mouth 3.5%, nose 1.3%
- PC5: mouth 5.7%, nose 0.8%

### 结论

1. **基向量主要捕捉运动语义而非身份**：
   - 所有数据集的PC1 dominant region均为mouth
   - 身份信息没有混入基向量（否则会有非运动区域异常激活）

2. **TT运动模式更多样**：
   - X模态PC1能量从IMR的64.6%降到TT的47.0%
   - TT的PC3出现eyehole dominant（水平眨眼运动）

3. **IMR垂直运动更一致**：
   - Y模态PC1能量：IMR 78.3% > TT 69.1%
   - 说明IMR被试的垂直张嘴模式更相似

4. **能量分布对比**：

| 数据集 | X PC1 | X PC1+PC2 | Y PC1 | Y PC1+PC2 |
|--------|-------|-----------|-------|-----------|
| IMR | 64.6% | 89.7% | 78.3% | 93.2% |
| TT | 47.0% | 74.2% | 69.1% | 87.5% |

TT的前两个基能量更分散，说明TT数据包含更丰富的运动变化。

---

## Idea 3 验证：差分 vs 非差分距离矩阵对比（2026-03-29）

> **脚本**: `scripts/svd_single_patient_raw.py` (单患者) / `scripts/svd_multi_patient_raw.py` (多患者)
> **结果目录**: `data/win20-step20/svd_single_patient_raw_results/`

### 实验设计

对比**原始距离矩阵（非差分）** vs **差分距离矩阵**的SVD分解结果，验证差分步骤的必要性。

- 数据：同10个患者（9 IMR + 1 TT）
- 方法：对原始距离矩阵直接做SVD（跳过差分步骤）
- 分析：PC1能量占比、语义区域分布

### 关键结果对比

| 指标 | RAW (非差分) | DIFF (差分) |
|------|-------------|-------------|
| **X PC1 dominant** | **eyehole** (100%患者) | **mouth** (100%患者) |
| **Y PC1 dominant** | mouth/around_mouth (分散) | **mouth** (100%患者) |
| **PC1 能量占比** | ~99.9% | ~95% |

### 详细结果

**RAW矩阵 (非差分):**
- X模态: PC1 dominant = eyehole (眼窝区域)
- Y模态: PC1 dominant = mouth/around_mouth (分散)
- PC1能量几乎100%，说明"人脸结构"主导了这个矩阵

**DIFF矩阵 (差分):**
- X模态: PC1 dominant = mouth (嘴部区域)
- Y模态: PC1 dominant = mouth (100%患者一致)
- PC1能量约95%，能量更分散到其他基

### 结论

1. **RAW矩阵的PC1捕捉的是静态结构**：eyehole dominant反映的是人脸解剖结构的差异（身份信息），而非运动

2. **差分是捕捉运动的必要步骤**：不做差分则PC1 dominant是eyehole（身份），而非mouth（运动）

3. **对后续方向的影响**：
   - ❌ 不需要做"坐标 vs 距离"对比了（差分距离矩阵已验证是正确方向）
   - ✅ 直接进入**Idea 2 (Tucker分解)**：验证多线性结构能否进一步解耦身份-运动

### 下一步方向

1. **Tucker分解验证**：多线性结构可能实现身份-运动的显式解耦
2. **TT场景复杂性分析**：为何TT能量更分散？是否与采集环境、患者异质性相关？
3. **时间系数可解释性**：按患者分段显示时间系数，验证是否反映运动阶段
4. **基向量语义映射**：学习从基向量到AU标签的线性映射

---

## Idea 4 验证：Grassmann流形验证共享基（2026-03-30）

> **脚本**: `scripts/grassmann_cross_analysis.py`
> **结果目录**: `data/win20-step20/grassmann_cross_analysis_results/`

### 问题

联合SVD是否"强制"共享基？单患者SVD与联合SVD的基是否真的相似？

### 验证方法

使用Grassmann流形主角度分析：
1. 对每个患者单独做SVD，得到各自的基
2. 对每个数据集（IMR/TT）做联合SVD，得到数据集级别的基
3. 计算单患者基与联合基之间的主角度

**核心逻辑**：
- 若"真共享"：单患者基与本数据集联合基角度小，单患者基与异数据集联合基角度大
- 若"被计算强迫"：所有患者与所有联合基角度相近，无差异化

### 关键结果

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

### 结论

- ✅ **联合基本身很接近**：TT_joint与IMR_joint夹角仅13.5°(X)和7.1°(Y)，两数据集共享相似的运动子空间
- ✅ **患者与本数据集联合基对齐更好**：IMR患者→IMR联合(12.9°) < IMR患者→TT联合(20.0°)；TT患者→TT联合(14.8°) < TT患者→IMR联合(20.0°)
- ✅ **跨数据集差异主要由数据集内部异质性驱动**，而非系统偏差
- ✅ **共享基不是被强迫的**：单患者SVD与联合SVD在 Grassmann 流形上确实自然对齐

---

## Idea 1* 补充：DMD动态模式分解（2026-03-31）

> **背景**: SVD窗口过粗（20帧≈状态表征），无法捕捉运动"过程"。win5-step5（5帧窗口，5帧步长≈过程表征）可补充过程信息。
>
> **目标**: 用DMD（动态模式分解）分析win5-step5数据，验证是否能捕捉到更细粒度的运动动态。
>
> **脚本**: `scripts/dmd_single_patient.py` (单患者), `scripts/dmd_multi_patient.py` (多患者联合)
> **结果目录**: `data/win5-step5/dmd_single_patient_results/`, `data/win5-step5/dmd_multi_patient_results/`

### DMD vs SVD 对比

| 特性 | SVD | DMD |
|------|-----|-----|
| 分解目标 | 矩阵最优低秩近似 | 线性动力系统拟合 |
| 输出 | 正交基 + 系数 | 动态模态 + 特征值（与动力学频率/衰减相关） |
| 语义 | 能量最大方向 | 动态模式（可解释为振荡/衰减） |
| 窗口 | 20帧静态窗口 | 可跨窗口时序建模 |

### 分析计划

#### 阶段1: DMD单患者验证 ✅ (2026-03-31)
1. 对每个患者单独做DMD
2. 分析模态数量、特征值分布（实部=衰减，虚部=振荡频率）
3. 可视化 dominant DMD modes → reshape成341×341热图
4. **对比**: DMD模态热图 vs SVD PC热图

**关键结果**:
- **所有10个患者 Mode1 dominant region = mouth** (与SVD一致)
- **特征值模 < 1** (X均值0.778, Y均值0.739) → 运动随时间衰减
- **存在复数特征值** → 振荡运动分量（如眼球运动）
- **TT患者窗口数更多** (TT/845380: 81窗口 vs IMR: ~26窗口)

| 患者 | X Mode1 dominant | X 特征值模 | Y Mode1 dominant | Y 特征值模 |
|------|-----------------|-----------|-----------------|-----------|
| IMR/00235 | mouth (9.4%) | 0.926 | mouth (6.4%) | 0.983 |
| IMR/00370 | mouth (5.0%) | 0.896 | mouth (7.9%) | 0.983 |
| IMR/00241 | mouth (8.4%) | 0.928 | mouth (8.2%) | 0.982 |
| IMR/00350 | mouth (9.5%) | 0.923 | mouth (7.6%) | 0.978 |
| IMR/00381 | mouth (4.3%) | 0.921 | mouth (3.6%) | 0.980 |
| TT/860312 | mouth (8.1%) | 0.925 | mouth (1.2%) | 0.981 |
| TT/862049 | mouth (7.9%) | 0.927 | mouth (4.0%) | 0.984 |
| TT/845380 | mouth (8.0%) | 0.937 | mouth (2.0%) | 0.986 |
| TT/846108 | mouth (6.4%) | 0.932 | mouth (2.4%) | 0.983 |
| TT/859088 | mouth (5.2%) | 0.941 | mouth (3.2%) | 0.983 |

#### 阶段2: DMD多患者联合分析 ✅ (2026-03-31)
- 筛选条件: 30 ≤ 窗口数 ≤ 200
- DMD rank = 50
- 结果目录: `data/win5-step5/dmd_multi_patient_results/`
- 模态保存: `saved_modes/{IMR,TT}/`

**筛选结果**:
| 数据集 | 筛选前患者 | 筛选后患者 | 总窗口数 |
|--------|-----------|-----------|---------|
| IMR | 227 | 2 | 210 |
| TT | 58 | 38 | 2119 |

**IMR联合DMD结果** (仅2患者，样本较少):

| 模态 | X 特征值 | X \|λ\| | X dominant | Y 特征值 | Y \|λ\| | Y dominant |
|------|---------|---------|------------|---------|---------|------------|
| Mode1 | 0.40+0.63i | 0.744 | mouth (7.8%) | 0.39-0.67i | 0.778 | mouth (7.8%) |
| Mode2 | 0.40-0.63i | 0.744 | mouth (7.8%) | 0.39+0.67i | 0.778 | mouth (7.8%) |
| Mode3 | 0.59+0.42i | 0.724 | mouth (9.9%) | -0.10-0.73i | 0.741 | mouth (1.8%) |
| Mode4 | 0.59-0.42i | 0.724 | mouth (9.9%) | -0.10+0.73i | 0.741 | mouth (1.8%) |

**TT联合DMD结果** (38患者，2119窗口):

| 模态 | X 特征值 | X \|λ\| | X dominant | Y 特征值 | Y \|λ\| | Y dominant |
|------|---------|---------|------------|---------|---------|------------|
| Mode1 | 0.584+0.00i | 0.584 | mouth (8.9%) | 0.453+0.00i | 0.453 | mouth (7.7%) |
| Mode2 | -0.006+0.54i | 0.543 | eyehole (1.8%) | 0.18+0.37i | 0.408 | mouth (1.6%) |
| Mode3 | -0.006-0.54i | 0.543 | eyehole (1.8%) | 0.18-0.37i | 0.408 | mouth (1.6%) |

**关键发现**:
- ✅ **Mode1 dominant region = mouth** (与SVD一致)
- ✅ **TT Mode2出现eyehole** (类似眨眼振荡模式)
- ✅ **特征值模 < 1** (所有运动都在衰减)

#### 阶段3: DMD模态与Blendshape相关性分析 ✅ (2026-04-01)
使用TT联合DMD的5个主模态，对所有患者（IMR+TT）计算时间系数，然后与blendshape做相关性分析

**blendshape窗口**: 5帧（与DMD差分窗口一致）

**关键结果**:

| DMD模态 | 最高相关blendshape | r | 特征值 |
|---------|-------------------|-----|--------|
| X Mode1 | jawForward | 0.589 | 0.573 |
| X Mode2 | eyeBlinkLeft | 0.317 | -0.008+0.53i |
| Y Mode1 | cheekPuff | 0.599 | 0.440 |
| Y Mode2 | cheekPuff | 0.636 | 0.380+0.09i |

**发现**:
- Y模态与 **cheekPuff** 相关性最高 (~0.6)
- X模态与 **jawForward** 相关性较高 (~0.58)
- 复数特征值模态对(2,3)和(4,5)的相关性完全一致（conjugate pairs）
- 与SVD的blendshape相关性结果（cheekPuff r=0.73）可比

#### 阶段4: Grassmann验证（可选）
- 类似Idea 4，验证DMD联合基是否也是"真共享"

#### 阶段4: 与SVD互补性分析
- 对同一患者，分别用SVD(win20)和DMD(win5)提取基
- 对比模态热图：SVD捕捉"状态"，DMD捕捉"过程"
- 验证两者是否捕捉不同频谱的运动信息

### 预期输出

| 输出 | 内容 |
|------|------|
| DMD模态热图 | 前5个dominant模态reshape为341×341热图 |
| 特征值分布 | 复平面上的特征值分布图 |
| 能量对比表 | DMD vs SVD各模态能量占比 |
| 时序系数 | 每个窗口在DMD模态上的激活系数 |

### 待解决问题

1. **DMD参数选择**: 如何确定截断的模态数量？
2. **时序对齐**: win5-step5窗口数远多于win20-step20，如何对应blendshape标签？
3. **初始化对比**: DMD能否解释SVD的PC1 dominant region？

---

## Idea 5 分析：码本基的类型分析（2026-04-12）

> **脚本目录**: `scripts/val_codebook/`
> **数据来源**: `data/win20-step20/IMR-SVD/` 和 `TT-SVD/`
> **结果目录**: `scripts/val_codebook/{exp1,exp2,exp3}_*/output/`

### 问题

单患者SVD提取的PC1主模态是否可以作为"公共码本"，用于区分患者的不同属性？

### 实验设计

**数据**: 269患者 (IMR=227, TT=42)
**特征**: 每个患者的PC1空间基 (341×341) 展平为116281维向量
**矩阵类型**:
- Full: 完整341×341矩阵
- Mouth: 截断的119×119矩阵 (around_mouth + mouth, indices 188-307)

**算法**: PCA(50) + Logistic Regression (L2)
**验证**: 5折分层交叉验证

### 实验1: IMR vs TT 数据集分类 (二分类)

**目的**: 验证IMR和TT患者的PC1模态是否存在差异

| Config | Accuracy | F1 (macro) | AUC |
|--------|----------|------------|-----|
| full_x | **0.918±0.040** | 0.829±0.088 | **0.962±0.027** |
| full_y | 0.914±0.038 | **0.835±0.080** | 0.882±0.086 |
| mouth_x | 0.903±0.027 | 0.813±0.048 | 0.909±0.040 |
| mouth_y | 0.881±0.040 | 0.783±0.068 | 0.921±0.030 |

**结论**:
- IMR和TT数据集在PC1模态上**差异显著**，AUC达0.96
- X方向AUC更高(0.962)，说明水平方向差异更大
- Mouth截断矩阵效果略差于full矩阵

### 实验2: 测别分类 (Left/Normal/Right, 三分类)

**目的**: 验证患者面瘫侧别（左侧/正常/右侧）在PC1模态上是否存在差异

**标签映射**:
- label_5class < 2 → Left (左侧异常)
- label_5class = 2 → Normal (正常)
- label_5class > 2 → Right (右侧异常)

**类别分布**: Left=78, Normal=97, Right=94

| Config | Accuracy | F1 (macro) |
|--------|----------|------------|
| full_x | 0.732±0.039 | 0.732±0.037 |
| full_y | 0.751±0.044 | 0.750±0.048 |
| **mouth_x** | **0.762±0.048** | **0.760±0.049** |
| mouth_y | 0.691±0.029 | 0.692±0.030 |

**结论**:
- PC1模态包含**左右不对称的特征**，三分类准确率约73-76%
- Mouth区域在X方向表现最好(76.2%)
- Y方向full表现更好(75.1%)

### 实验3: 严重度分类 (Normal/Mild/Severe, 三分类)

**目的**: 验证面瘫严重程度在PC1模态上是否存在差异

**标签映射**:
- score = 0 → Normal
- score = 1 → Mild
- score = 2 → Severe

**类别分布**: Normal=97, Mild=70, Severe=102

| Config | Accuracy | F1 (macro) |
|--------|----------|------------|
| full_x | 0.584±0.033 | 0.563±0.044 |
| full_y | 0.594±0.069 | 0.563±0.077 |
| **mouth_x** | **0.613±0.053** | **0.595±0.058** |
| mouth_y | 0.577±0.068 | 0.564±0.067 |

**结论**:
- 面瘫等级在PC1模态上**区分度较低**（约58-61%）
- 接近随机基线(33%三类)，说明PC1难以捕捉严重程度信息
- Mouth区域X方向仍然最好(61.3%)

### 关键结论

1. **数据集效应显著**: IMR vs TT分类准确率91.8%，AUC=0.96
   - 提示跨数据集建模时需考虑数据集偏差

2. **Mouth区域是主要判别区域**:
   - 测别分类: Mouth_X最好(76.2% vs 73.2% full)
   - 严重度分类: Mouth_X最好(61.3% vs 58.4% full)

3. **X方向普遍比Y方向效果更好**:
   - IMR vs TT: X方向AUC=0.962
   - 测别分类: Mouth_X=76.2%
   - 严重度分类: Mouth_X=61.3%

4. **严重度分类困难**:
   - 严重度分类准确率仅58-61%，接近随机(33%)
   - 说明单患者PC1难以捕捉严重程度信息

5. **PC1码本性质**:
   - PC1可以区分数据集和侧别(Left/Right)
   - PC1不能区分严重程度
   - PC1主要包含运动方向信息，而非强度信息

### 代码结构

```
scripts/val_codebook/
├── README.md                    # 任务说明
├── RESULTS.md                   # 结果汇总
├── common/
│   ├── load_pc1.py            # 数据加载
│   ├── classify.py              # PCA+LR分类框架
│   └── visualize.py            # ROC/混淆矩阵可视化
├── exp1_dataset_classification/output/
│   ├── roc_*.png, confusion_*.png
│   └── results_*.json
├── exp2_side_classification/output/
├── exp3_severity_classification/output/
└── sweep.py                    # 批量运行脚本
```
