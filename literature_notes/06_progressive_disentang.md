# 文献笔记：Progressive Disentangled Talking Head

> **论文**: Progressive Disentangled Representation Learning for Fine-Grained Controllable Talking Head Synthesis
> **作者**: Duomin Wang et al.
> **年份**: 2022
> **arXiv**: 2211.14506
> **方向标签**: #talking-head #fine-grained-control #progressive-disentanglement #lip-eye-pose-emotion

---

## 一句话核心思想

**粗到细的渐进式解耦学习**：先从驱动信号提取统一运动特征，再逐步分离出唇动、眼动/眨眼、头部姿态、表情四个细粒度因子。

---

## 方法：渐进式解耦

```
Step 1: 统一运动特征（粗粒度）
         ↓
Step 2: 唇动（细粒度，对比学习）
         ↓
Step 3: 眼动/眨眼（对比回归）
         ↓
Step 4: 头部姿态（特征解相关）
         ↓
Step 5: 表情（自重建）
```

### 各因子解耦策略

| 因子 | 解耦策略 |
|------|---------|
| 唇动 | 对比学习 + 回归 |
| 眼动/眨眼 | 对比回归 |
| 头部姿态 | 特征级解相关 |
| 表情 | 自重建 |

---

## 与你的方向对比

**借鉴点**: 渐进式解耦思路 — 你的子运动分解也可以从粗到细，先分解大类再细分

---

## 笔记时间

2026-03-18
