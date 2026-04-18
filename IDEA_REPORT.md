# Research Idea Report

[TOC]

**Direction**: 基于关键点距离时序矩阵的面部子运动单元分解 — 使用矩阵分解方法(NMF、Tucker分解、CP分解)将面部关键点距离矩阵分解为可解释的子运动基，实现面部运动的细粒度分解与重建

**Final Target**：基于一组可能存在面瘫(表现为嘴部不对称)的患者，利用其分解得到的面部子单元，系数，和分解残余量，实现面瘫分级

**Data**: 健康组被试完成指定咧嘴任务，患病组完成指定咧嘴任务

**Generated**: 2026-03-19

**Ideas evaluated**: 10 generated → 5 surviving filtering → 0 piloted → 5 recommended

---

## Landscape Summary

### 现有方法分析

| 方法 | 年份 | 核心思路 | 运动分解方式 | 缺点 |
|------|------|---------|-------------|------|
| Sparse Facial Motion | 2025 | VQ-VAE离散关键帧+过渡帧 | 关键帧=子单元 | 离散化，预测任务 |
| AniTalker | 2024 | MINE互信息解耦身份vs运动 | 运动编码器 | 需要大量视频训练 |
| EDTalk | 2024 | 三空间正交基向量 | 口型/姿态/表情分离 | 基数量需手动设定 |
| MoDiTalker | 2024 | 两阶段AToM+MToV | 唇动与视频分离 | 粒度粗，只解耦唇动 |
| DisCoHead | CVPR 2023 | 几何变换瓶颈 | 头姿与外观分离 | 几何变换局限 |
| Progressive | CVPR 2022 | 粗到细渐进解耦 | 唇动/眼动/头姿/表情 | 对比学习需负样本 |
| **Blendshapes** | **2025** | **语音+表情blendshape叠加** | **线性加性分解** | **只有两类，粒度粗** |
| FacialMotionID | 2025 | 实证研究 | 证明运动可辨识身份 | 非方法论论文 |

### 已识别的研究空白

1. **矩阵分解视角缺失**: 所有方法都是神经网络隐式解耦，没有显式MF方法
2. **关键点距离矩阵未利用**: 没有方法用距离矩阵作为输入
3. **连续可加表示**: 都是离散token，没有连续的子运动叠加
4. **可解释性**: 基数量需手动设定，不够自动
5. **身份解耦**: 用MINE深度学习方法，没有简单MF方法

---

## Recommended Ideas (ranked)

### Idea 1: Baseline验证 — 纯NMF能否发现可解释的面部子运动？

- **Hypothesis**: 即使不加任何先验知识，NMF从关键点距离矩阵中也能自动浮现出类似"唇动"、"眨眼"、"头动"等语义可解释的基向量

- **Minimum experiment**: 对478个关键点构建距离时序矩阵，直接应用标准NMF，可视化每个基的关键点对权重热图

- **Expected outcome**: 如果基向量具有语义可解释性 → 证明MF视角可行；如果基向量语义混杂 → 说明需要额外约束

- **Novelty**: 8/10 — 首次将标准NMF直接用于关键点距离矩阵

- **Feasibility**: 计算: 1GPU小时；数据: 任意公开人脸数据集；实现: sklearn 5行代码

- **Risk**: LOW（失败也有价值，明确回答方向是否可行）

- **Contribution type**: 诊断性

- **Pilot result**: SKIPPED — 建议手动快速验证

- **Reviewer's likely objection**: "标准NMF太简单，baseline而已，不算贡献"

- **Why we should do this**: 最低成本回答"MF视角是否可行"这个根本问题，结果无论正负都有意义

---

### Idea 2: Tucker分解实现身份-运动显式解耦

- **Hypothesis**: 不同人的同一种面部子运动在关键点距离空间中共享相同的"运动基"，但具有不同的"身份系数"，Tucker分解的多线性结构可直接分离这两者

- **Minimum experiment**: 收集10-20人视频，构建3D张量 X ∈ R^(N_subjects × N_pairs × T)，应用Tucker分解，验证同一人在身份基上系数相似，同一运动在不同人中共享运动基

- **Expected outcome**: 如果Tucker可分离 → 提出新的身份-运动解耦方法；如果分离不明显 → 说明需要更强的约束

- **Novelty**: 9/10 — 首次将Tucker分解用于面部运动-身份解耦

- **Feasibility**: 计算: 1-2 GPU小时；数据: 需要多人口型数据集；实现: 需自定义优化

- **Risk**: MEDIUM（身份基和运动基可能仍然混杂）

- **Contribution type**: 新方法

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "Tucker分解是否真的优于神经网络隐式解耦？需要和AniTalker等方法对比"

- **Why we should do this**: 最接近论文创新点，如果成功可直接对比AniTalker的隐式解耦

---

### Idea 3: 验证距离表示是否真的优于坐标表示（负面假设验证）

- **Hypothesis**: 距离矩阵可能丢失了关键信息（如整体平移），导致分解质量反而更差

- **Minimum experiment**: 对同一数据构建坐标矩阵 vs 距离矩阵，分别做NMF，对比重建误差、可解释性、泛化能力

- **Expected outcome**: 如果距离矩阵更差 → 重新考虑输入表示；如果等价或更好 → 证明距离表示的优势

- **Novelty**: 7/10 — 系统性对比两种表示的MF适用性

- **Feasibility**: 计算: 半天；数据: 任意数据集；实现: 标准NMF

- **Risk**: LOW（无论结果如何都有明确结论）

- **Contribution type**: 理论/诊断结果

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "这更像消融实验，不是核心贡献"

- **Why we should do this**: 回答"距离表示是否值得使用"这个前提问题，理论价值高

---

### Idea 4: 从分解基到可解释属性的映射学习

- **Hypothesis**: NMF/Tucker分解产生的基向量虽然可能语义混杂，但通过监督学习可以找到它们与已知语义标签（AU、blendshapes）之间的线性映射

- **Minimum experiment**: 使用带AU标注的数据集，对距离矩阵做NMF得到H，训练线性回归 AU_labels = M × H，检验哪些基对应哪些AU

- **Expected outcome**: 如果映射成功且稀疏 → 说明MF基具有语义可解释性；如果映射后基仍混杂 → 说明MF基与语义标签存在语义鸿沟

- **Novelty**: 7/10 — "可解释性后处理"的MF框架

- **Feasibility**: 计算: 1 GPU小时；数据: 需要AU标注数据集；实现: sklearn线性回归

- **Risk**: LOW（监督学习部分几乎一定成功）

- **Contribution type**: 经验发现

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "为什么不用有监督的分解方法？"

- **Why we should do this**: 快速验证可解释性，且可为后续监督约束提供依据

---

### Idea 5: 与SOTA深度解耦方法的公平对比

- **Hypothesis**: 神经网络端到端优化可能在重建质量上优于MF，但MF在可解释性和身份解耦上可能具有独特优势

- **Minimum experiment**: 获取EDTalk/AniTalker的解耦结果，在相同数据上运行NMF，对比: 重建质量、FaceNet身份识别准确率、AU回归可解释性

- **Expected outcome**: 如果MF接近深度方法 → 说明简单MF值得探索；如果差距大 → 说明需要结合深度学习

- **Novelty**: 6/10 — 为领域提供对比基线

- **Feasibility**: 计算: 半天；数据: 相同数据集；实现: 需要复现/获取对比方法

- **Risk**: LOW（对比实验设计清晰）

- **Contribution type**: 经验发现/对比基线

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "对比已有SOTA不是新贡献，需要超越它们"

- **Why we should do this**: 为后续研究提供公平比较的基线，MF方法的必要前提

---

## Eliminated Ideas (for reference)

| Idea | Reason eliminated |
|------|-------------------|
| Idea 6: 双视图联合分解 | 实现复杂，收益不确定 |
| Idea 7: 时间连续性约束 | 需要自定义优化，验证周期长 |
| Idea 8: 贝叶斯NMF自动基数量 | Beta-NMF对超参数敏感，可能推断不稳定 |
| Idea 9: 交互式MF工具 | 工具性质，非核心论文贡献 |
| Idea 10: 跨数据集迁移性 | 依赖Idea 1/2成功 |

---

## Suggested Execution Order

1. **先做Idea 1（1-2天）**: 快速验证MF视角是否可行
2. **同时做Idea 3（3-5天）**: 验证距离表示是否合理
3. **根据Idea 1/3结果**:
   - 如果正面 → 深入Idea 2（Tucker分解）
   - 如果负面 → 考虑结合深度学习（Idea 9）
4. **Idea 4和Idea 5作为辅助验证**

---

## Key Open Questions

1. **NMF基是否具有语义可解释性？** — Idea 1回答
2. **距离矩阵是否比坐标矩阵更适合MF？** — Idea 3回答
3. **Tucker能否做到身份-运动显式解耦？** — Idea 2回答
4. **MF方法与深度解耦方法差距多大？** — Idea 5回答

---

## Next Steps

- [x] **Idea 1 ✅**: SVD差分验证完成 — PC1 mouth-dominant，运动语义可解释（详见 IDEA_EXPERIMENTS.md）
- [x] **Idea 3 ✅**: 差分必要性验证完成 — RAW=eyehole(身份)，DIFF=mouth(运动)（详见 IDEA_EXPERIMENTS.md）
- [x] **Idea 4 ✅**: Grassmann验证完成 — 共享基是"真共享"而非"计算强迫"（详见 IDEA_EXPERIMENTS.md）
- [x] **Idea 2 ❌**: Tucker不适合可视化需求
- [ ] PARAFAC/CP分解探索（替代Tucker）
- [ ] 结合blendshape/AU标注进行弱监督语义映射
- [ ] 时间系数可解释性分析
