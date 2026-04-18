# 文献笔记：DisCoHead

> **论文**: DisCoHead: Audio-and-Video-Driven Talking Head Generation by Disentangled Control of Head Pose and Facial Expressions
> **作者**: Geumbyeol Hwang et al.
> **年份**: 2023
> **arXiv**: 2303.07697
> **方向标签**: #talking-head #pose-expression-disentanglement #geometric-transformation

---

## 一句话核心思想

用**单一几何变换**（仿射变换或薄板样条变换）作为瓶颈，从驱动视频中解耦并提取头部运动，然后用音频控制口型、另一个视频控制眼部。

---

## 方法核心

```
驱动视频 → 几何变换瓶颈 → 头部运动（解耦）
                        ↓
                源身份图像 + 头部运动
                        ↓
                口型控制（音频）+ 眼部控制（另一视频）
                        ↓
                    说话人脸
```

### 几何变换瓶颈

- 几何变换（仿射/薄板样条）天然将头部运动从外观中分离
- 无监督，不需要标注

---

## 与你的方向对比

**借鉴点**: 用**几何/参数空间**的瓶颈做解耦，而非直接在高维图像空间解耦 — 你的矩阵分解框架也属于参数空间的分解

---

## 笔记时间

2026-03-18
