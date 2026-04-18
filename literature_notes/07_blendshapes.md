# 文献笔记：Disentangled Speech- and Expression-Driven Blendshapes

> **论文**: Learning Disentangled Speech- and Expression-Driven Blendshapes for 3D Talking Face Animation
> **作者**: Yuxiang Mao et al.
> **年份**: 2025
> **arXiv**: 2510.25234
> **方向标签**: #blendshapes #speech-expression-disentanglement #3D-face #sparsity

---

## 一句话核心思想

将3D面部动画的运动驱动分解为**语音驱动blendshape**和**表情驱动blendshape**两组，引入稀疏约束损失鼓励两者解耦，同时允许捕获跨域变形。

---

## 方法

```
VOCAset (语音驱动的3D面部数据) + Florence4D (表情序列数据)
                    ↓
         联合学习两组blendshape
    ┌──────────────┴──────────────┐
    ↓                              ↓
语音驱动的blendshape        表情驱动的blendshape
    ↓                              ↓
    └──────────┬──────────────────┘
               ↓
        稀疏约束损失
    （鼓励解耦，同时允许跨域变形）
               ↓
        FLAME模型参数
      （表情参数 + jaw pose）
               ↓
        3D Gaussian avatars动画
```

---

## 与你的方向对比

| 你的方向 | 这篇 |
|---------|------|
| 整体运动 → 子运动 | ✅ 运动 = 语音blendshape + 表情blendshape 的线性叠加 |
| 线性可加性 | ✅ 明确的加性组合假设 |

**最相关的论文**: 这篇的"blendshape = 子运动基，线性叠加 = 整体运动"与你的矩阵分解思路**最接近**

---

## 笔记时间

2026-03-18
