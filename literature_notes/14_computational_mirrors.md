---
title: "Computational Mirrors: Blind Inverse Light Transport by Deep Matrix Factorization"
authors: ["Miika Aittala", "Prafull Sharma", "Lukas Murmann", "Adam B. Yedidia", "Gregory W. Wornell", "William T. Freeman", "Fredo Durand"]
year: 2019
venue: NeurIPS
arxiv_id: 1912.02314
categories: ["cs.CV", "cs.LG"]
abstract: |
  We recover a video of the motion taking place in a hidden scene by observing changes
  in indirect illumination in a nearby uncalibrated visible region. We solve this problem
  by factoring the observed video into a matrix product between the unknown hidden scene
  video and an unknown light transport matrix. This task is extremely ill-posed, as any
  non-negative factorization will satisfy the data. Inspired by recent work on the Deep
  Image Prior, we parameterize the factor matrices using randomly initialized convolutional
  neural networks trained in a one-off manner, and show that this results in decompositions
  that reflect the true motion in the hidden scene.
url: https://arxiv.org/abs/1912.02314
pdf: https://arxiv.org/pdf/1912.02314.pdf

---

## 方向

**方向三.1：深度矩阵分解（最相关的参考工作）**

## 核心贡献

1. **深度矩阵分解**：将观测视频分解为隐藏场景视频 × 光传输矩阵的矩阵乘积
2. 使用随机初始化的CNN参数化因子矩阵（Deep Image Prior思想）
3. 一次性训练，避免过拟合

## 方法细节

- 问题极度病态：任何非负分解都满足数据
- 解决方案：使用随机初始化CNN参数化因子矩阵
- Deep Image Prior的启发：卷积网络的结构和随机初始化提供了一个好的先验
- 单次训练（one-off），无需迭代优化

## 与本研究的关联

- **⭐⭐⭐⭐ 核心参考**：本研究也是将矩阵分解应用于运动分析
- 关键区别：本文是盲分解（blind factorization），而你的研究是有监督的（使用SVD/DMD已知基）
- **深度矩阵分解范式**：用神经网络参数化因子矩阵，可以扩展到非线性分解
- 可借鉴的思想：用CNN参数化基矩阵B，而非使用固定的正交基

## 关键词

Deep Matrix Factorization, Blind Inverse Problem, Light Transport, Deep Image Prior, Convolutional Neural Networks
