# 面部运动解耦与分解文献时间线梳理

**日期**: 2026-03-29
**目的**: 梳理面部运动解耦/分解领域的发展脉络，便于后续做路线图

[TOC]

---

## 一、按时间线的方法演进

### 2000s: 经典方法时代

| 年份 | 方法 | 代表工作 | 核心特点 |
|------|------|---------|---------|
| 2004 | NMF面部应用 | Application of non-negative matrix factorization | NMF首次用于人脸 |
| 2007 | Projective NMF | Projective Non-Negative Matrix Factorization | NMF用于面部表情 |
| 2009 | Graph-preserving Sparse NMF | Facial expression recognition based on graph-preserving | 稀疏约束NMF |
| 2010 | Graph-Preserving Sparse NMF | Graph-Preserving Sparse Nonnegative Matrix Factorization | 图约束NMF |
| 2012 | Discriminant NMF | Subclass discriminant Non-negative Matrix Factorization | 判别NMF |
| 2012 | GPA + PCA | Popat - Lip movement statistical modelling | 统计建模唇运动 |

**特点**: 基于NMF/PCA的线性分解方法，监督或无监督，主要用于静态人脸/表情识别

---

### 2015-2019: 经典方法的深化 + 张量分解兴起

| 年份 | 方法 | 代表工作 | 核心特点 |
|------|------|---------|---------|
| 2015 | Extended NMF | Extended Non-negative Matrix Factorization | NMF扩展到表情识别 |
| 2016 | 光流法 | Initial assessment of facial nerve paralysis (26578273) | 面神经麻痹运动分析 |
| 2017 | **帧间距离 + Procrustes** | **Hallac - Dynamic facial asymmetry (28011182)** | **差分距离概念** |
| 2018 | Sparse Orthogonal Tucker | Sparse Orthogonal Tucker for 2D+3D FER | Tucker用于表情识别 |
| 2019 | Low-rank Tucker | Low-rank Tucker for 2D+3D FER | 低秩Tucker |
| 2019 | Fluctuating asymmetry | Khambay - Dynamic smiles asymmetry (30940397) | 动态微笑不对称 |
| 2019 | Orthogonal Low-rank Tucker | Orthogonal Low Rank Tucker for FER | 正交+低秩Tucker |

**特点**:
- Tucker/张量分解开始用于面部
- 运动分析从静态转向动态
- Hallac 2017提出帧间距离思路

---

### 2020-2022: 深度学习崛起前夕

| 年份 | 方法 | 代表工作 | 核心特点 |
|------|------|---------|---------|
| 2020 | Sparse+Low-rank Tucker | Sparse and Low-rank Tucker for 2D+3D FER | 稀疏+低秩Tucker |
| 2020 | Kinematic Analysis | Hallac - Kinematic Analysis of Smiles (31726862) | 3D运动学分析 |
| 2021 | Low-rank Graph Tucker | Low Rank Graph Regularization Tucker | 图正则Tucker |
| 2021 | 3D Motion Capture | Zhao/Hallac - Cleft lip motion capture (33770029) | 3D运动捕捉 |
| 2021 | **FaceFormer** | FaceFormer: Speech-Driven 3D Facial Animation (CVPR 2022) | Transformer-based |
| 2022 | Low Rank Tucker | Low Rank Tucker for 2D+3D FER (2022) | 低秩Tucker |
| 2022 | AU-based | Gaber - Facial paralysis AU-based (36516130) | **基于AU的面瘫评估** |
| 2022 | **DisCoHead** | DisCoHead: Audio-driven talking head (CVPR 2023) | 几何变换解耦 |
| 2022 | Emotion-Controllable Talking Face | Emotion-Controllable Generalized Talking Face Generation | 情感可控 |

**特点**:
- Tucker在面部表情识别中持续深化
- **FaceFormer引入Transformer到talking face**
- AU-based方法开始用于面瘫评估

---

### 2023-2024: 深度学习解耦爆发期

| 年份 | 方法 | 代表工作 | 核心特点 |
|------|------|---------|---------|
| 2023 | **DisCoHead** | DisCoHead (CVPR 2023) | 几何变换瓶颈解耦 |
| 2024 | **AniTalker** | AniTalker (arXiv 2024) | MINE互信息解耦 |
| 2024 | **EDTalk** | EDTalk (arXiv 2024) | 三空间正交约束 |
| 2024 | **MoDiTalker** | MoDiTalker (arXiv 2024) | 扩散模型两阶段 |
| 2024 | 3D Topography | Hallac - Pediatric facial palsy (39476531) | 曲率分析+微笑曲线 |
| 2024 | Nasolabial dynamics | Patel - Nasolabial dynamics (37541045) | 鼻唇沟动态 |

**特点**:
- **所有talking face方法都转向深度学习**
- 身份-运动解耦成为核心问题
- 医学应用开始关注区域特异性不对称

---

### 2025-2026: 深度学习 + 医学评估深化期

| 年份 | 方法 | 代表工作 | 核心特点 |
|------|------|---------|---------|
| 2025 | **Sparse Facial Motion** | VQ-VAE + 稀疏关键帧 | 离散化+心理学对齐 |
| 2025 | **Blendshapes** | 语音+表情blendshape叠加 | 线性加性分解 |
| 2025 | **EDTalk++** | Full disentanglement (arXiv 2025) | 完全解耦 |
| 2025 | Automated grading | Knoedler - Automated palsy grading (39732200) | 自动面瘫分级 |
| 2026 | **Micro-Expression Disentanglement** | Learnable Feature Disentanglement with Temporal-Complemented Motion Enhancement (Entropy 2026) | 微表情识别+时序增强 |

**特点**:
- talking face领域继续深化解耦
- 医学评估开始自动化
- 出现blendshape线性叠加思想（与你的NMF/Tucker思路类似）
- 2026年出现结合时序增强的解耦方法

---

## 二、方法分类对比

### 2.1 分解方法对比

| 方法类别 | 代表工作 | 分解类型 | 解耦方式 | 可解释性 |
|---------|---------|---------|---------|---------|
| **NMF家族** | 2004-2015 | Parts-based分解 | 无显式解耦 | 高（部分可解释） |
| **Tucker/张量** | 2018-2022 | 多线性分解 | 无显式解耦 | 中（结构化） |
| **字典学习** | 2016-2019 | 稀疏表示 | 无显式解耦 | 中 |
| **Procrustes** | 2012-2021 | 形状比较 | 无 | 低（仅比较） |
| **深度学习** | 2022-2025 | 隐式分解 | 互信息/正交约束 | **低（黑盒）** |
| **几何变换** | 2023 | 分解瓶颈 | 几何变换 | 中 |
| **Blendshape** | 2025 | 线性叠加 | 无 | 高 |

### 2.2 应用场景对比

| 场景 | 代表方法 | 你的场景 |
|------|---------|---------|
| Talking face合成 | AniTalker, EDTalk, DisCoHead | ❌ 不同 |
| 面部表情识别 | Tucker FER, NMF FER | ❌ 不同 |
| 唇裂/面瘫评估 | Hallac系列, AU-based | ✅ **一致** |
| 面部运动分析 | Hallac帧间距离 | ✅ **一致** |

---

## 三、你的研究定位

### 3.1 时间线上的位置

```
2000s: NMF/PCA线性分解（无解耦）
    ↓
2010s: Tucker张量分解（无解耦） + 运动分析兴起
    ↓
2020s: 深度学习解耦爆发（talking face）
    ↓
2025+: 医学应用深化 + 线性分解思想回归？
        ↑
    你的研究：差分距离矩阵 + Tucker/NMF + 面瘫分级
```

### 3.2 核心Gap

| Gap | 现有工作 | 你的机会 |
|-----|---------|---------|
| **帧间距离→矩阵分解** | Hallac 2017计算帧间距离，仅做Procrustes比较 | **用SVD/Tucker分解差分矩阵** |
| **深度学习→经典方法** | 所有2022+的解耦都用深度学习 | **矩阵分解的显式可解释性** |
| **静态→动态距离矩阵** | NMF/Tucker用于静态图像 | **341×341时序差分距离矩阵** |
| **合成→医学评估** | 所有方法针对talking face合成 | **面瘫分级应用** |

---

## 四、关键文献汇总

### 4.1 最相关的经典方法文献

| 论文 | PMID | 年份 | 方法 | 为什么相关 |
|------|------|------|------|-----------|
| Hallac - Dynamic facial asymmetry | 28011182 | 2017 | 帧间距离+Procrustes | **帧间距离与你完全一致** |
| Hallac - Pediatric FP | 39476531 | 2024 | 曲率分析+微笑曲线 | 面瘫分级场景一致 |
| Gaber - AU-based FP | 36516130 | 2022 | AU监督评估 | 面瘫分级场景一致 |
| Knoedler - Automated palsy grading | 39732200 | 2025 | 自动分级 | 面瘫分级场景一致 |

### 4.2 最相关的深度学习文献

| 论文 | 年份 | 方法 | 为什么相关 |
|------|------|------|-----------|
| FaceFormer | 2022 | Transformer-based speech-driven | Transformer引入talking face |
| AniTalker | 2024 | MINE互信息解耦 | 身份-运动解耦思路一致 |
| EDTalk | 2024 | 三空间正交约束 | 基向量正交化可借鉴 |
| DisCoHead | 2023 | 几何变换解耦 | 几何角度的解耦思想 |
| Sparse Facial Motion | 2025 | VQ-VAE+稀疏 | 离散化+心理学对齐 |
| Micro-Expression 2026 | 2026 | 时序增强解耦 | 时序+解耦最新工作 |

### 4.3 Tucker/NMF基础文献

| 论文 | 年份 | 方法 | 为什么相关 |
|------|------|------|-----------|
| Sparse Orthogonal Tucker (2018) | 2018 | Tucker+稀疏 | Tucker用于面部 |
| Low-rank Tucker (2019-2022) | 2019-2022 | Tucker+低秩 | Tucker用于FER |
| Extended NMF (2015) | 2015 | NMF扩展 | NMF用于表情 |

---

## 五、总结

### 5.1 领域发展规律

1. **2000s**: NMF/PCA主导，强调可解释的Parts-based分解
2. **2010s**: Tucker/张量方法兴起，处理多维数据
3. **2017**: Hallac提出帧间距离思路（差分概念）
4. **2020s**: 深度学习完全主导，隐式解耦成为主流
5. **2025+**: 医学应用深化，Blendshape出现线性回归迹象

### 5.2 你的差异化机会

1. **方法论**: 经典矩阵分解 vs 深度学习（显式可解释性）
2. **输入表示**: 差分距离矩阵 vs 原始坐标/图像
3. **应用场景**: 面瘫分级 vs talking face合成
4. **解耦目标**: 身份-运动分离 vs 音频-视觉分离

---

**下一步**: 需要对2022-2026的深度学习解耦文献做更细致的调研
