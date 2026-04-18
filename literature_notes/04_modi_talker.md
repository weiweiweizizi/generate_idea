# 文献笔记：MoDiTalker

> **论文**: MoDiTalker: Motion-Disentangled Diffusion Model for High-Fidelity Talking Head Generation
> **作者**: Seyeon Kim et al.
> **年份**: 2024
> **arXiv**: 2403.19144
> **方向标签**: #talking-head #motion-disentanglement #diffusion #lip-sync

---

## 一句话核心思想

两阶段架构：**AToM** (Audio-to-Motion) 生成同步唇运动 + **MToV** (Motion-to-Video) 生成高质量头部视频，通过解耦运动与外观提升时序一致性。

---

## 方法

```
输入音频
   ↓
┌──────────────────┐
│ AToM (Audio-to-Motion) │
│ - 音频注意力机制捕获细微唇动 │ ← 音频信息
└──────────────────┘
   ↓ 唇运动
┌──────────────────┐
│ MToV (Motion-to-Video) │
│ - 高效三平面表示          │ ← 外观/头部信息
│ - 提升时序一致性          │
└──────────────────┘
   ↓
输出视频
```

---

## 核心贡献

- **AToM**: 音频注意力机制捕获细微唇动
- **MToV**: 三平面表示 + 时序一致性
- 扩散模型解决GAN的不稳定训练和时序不一致问题

---

## 与你的方向对比

**借鉴点**: 两阶段解耦（运动生成→视频渲染）的pipeline设计思路

---

## 笔记时间

2026-03-18
