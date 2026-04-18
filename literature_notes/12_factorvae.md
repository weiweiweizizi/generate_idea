---
title: "Disentangling by Factorising"
authors: ["Hyunjik Kim", "Andriy Mnih"]
year: 2018
venue: ICML
arxiv_id: 1802.05983
categories: ["stat.ML", "cs.LG"]
abstract: |
  We define and address the problem of unsupervised learning of disentangled representations
  on data generated from independent factors of variation. We propose FactorVAE, a method
  that disentangles by encouraging the distribution of representations to be factorial and
  hence independent across the dimensions. We show that it improves upon β-VAE by providing
  a better trade-off between disentanglement and reconstruction quality. Moreover, we
  highlight the problems of a commonly used disentanglement metric and introduce a new
  metric that does not suffer from them.
url: https://arxiv.org/abs/1802.05983
pdf: https://arxiv.org/pdf/1802.05983.pdf

---

## 方向

**方向二：带物理约束的自动编码器（解耦理论基础）**

## 核心贡献

1. **FactorVAE**：通过鼓励表示的边缘分布为阶乘（独立）来实现解耦
2. 提出了新的解耦度量指标（避免先前指标的 问题）
3. 在解耦质量和重建质量之间提供更好的权衡

## 方法细节

- 核心思想：使表示的分布是阶乘的（即各维度独立）
- 使用总相关性（Total Correlation）来衡量表示的独立性
- 相比β-VAE有更好的平衡

## 与本研究的关联

- **为IDEA_TO_REFINE.md中的"身份-运动解耦"提供理论基础**
- FactorVAE的阶乘分布思想可用于设计正交损失函数
- 解耦理论对于码本中各基向量的独立性约束有重要参考价值

## 关键词

Disentangled Representations, FactorVAE, Total Correlation, β-VAE, Unsupervised Learning
