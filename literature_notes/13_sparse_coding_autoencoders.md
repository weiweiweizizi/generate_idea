---
title: "Sparse Coding and Autoencoders"
authors: ["Akshay Rangamani", "Anirbit Mukherjee", "Amitabh Basu", "Tejaswini Ganapathy", "Ashish Arora", "Sang Chin", "Trac D. Tran"]
year: 2017
venue: arXiv
arxiv_id: 1708.03735
categories: ["cs.LG", "math.OC", "stat.ML"]
abstract: |
  In "Dictionary Learning" one tries to recover incoherent matrices A* in R^{n x h}
  (typically overcomplete and whose columns are assumed to be normalized) and sparse
  vectors x* in R^h with a small support of size h^p for some 0 < p < 1 while having
  access to observations y in R^n where y = A*x*. In this work we undertake a rigorous
  analysis of whether gradient descent on the squared loss of an autoencoder can solve
  the dictionary learning problem. The "Autoencoder" architecture we consider is a
  R^n -> R^n mapping with a single ReLU activation layer of size h.
url: https://arxiv.org/abs/1708.03735
pdf: https://arxiv.org/pdf/1708.03735.pdf

---

## 方向

**方向二：带物理约束的自动编码器（正交约束理论）**

## 核心贡献

1. 严格分析了自动编码器的梯度下降是否能解决字典学习问题
2. 证明了在稀疏码维度渐近情况下，标准平方损失函数的期望梯度在A*的小邻域内可以忽略
3. 揭示了ReLU层可以自动恢复稀疏码的支持集

## 方法细节

- 单层ReLU激活的自动编码器架构
- 分析了梯度下降的收敛行为
- 在合成数据上提供了实验验证

## 与本研究的关联

- **为IDEA_TO_REFINE.md中的Q3（正交性约束）提供理论参考**
- 字典学习与码本基的关系：码本可以看作连续字典的离散版本
- 稀疏性约束与正交性约束的对比：稀疏编码关注的是基的激活模式，而非基之间的正交性
- 对于理解码本基的性质提供了数学基础

## 关键词

Dictionary Learning, Sparse Coding, Autoencoder, Gradient Descent, ReLU
