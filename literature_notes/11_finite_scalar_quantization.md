---
title: "Finite Scalar Quantization: VQ-VAE Made Simple"
authors: ["Fabian Mentzer", "David Minnen", "Eirikur Agustsson", "Michael Tschannen"]
year: 2023
venue: NeurIPS
arxiv_id: 2309.15505
categories: ["cs.CV", "cs.LG"]
abstract: |
  We propose to replace vector quantization (VQ) in the latent representation of VQ-VAEs
  with a simple scheme termed finite scalar quantization (FSQ), where we project the VAE
  representation down to a few dimensions (typically less than 10). Each dimension is
  quantized to a small set of fixed values, leading to an (implicit) codebook given by
  the product of these sets. By appropriately choosing the number of dimensions and values
  each dimension can take, we obtain the same codebook size as in VQ.
url: https://arxiv.org/abs/2309.15505
pdf: https://arxiv.org/pdf/2309.15505.pdf

---

## 方向

**方向一：VQ-VAE码本失效与优化**

## 核心贡献

1. 用**标量量化**替代向量量化，避免codebook collapse
2. 投影到少量维度(≤10)，每维量化到固定值集合
3. 隐式codebook由这些集合的笛卡尔积给出

## 方法细节

- 选择合适的维度数量和每维的值数量，可以获得与VQ相同的codebook大小
- 训练更稳定，避免死基问题
- 避免了对codebook的复杂维护

## 与本研究的关联

- **解决IDEA_TO_REFINE.md中的Q1（码本冷启动）和Q2（梯度回传）问题**
- 可作为种子码本的替代方案
- FSQ的简单性使其更容易集成到端到端训练中

## 关键词

VQ-VAE, Finite Scalar Quantization, Codebook, Discretization
