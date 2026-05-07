# CLAUDE.md

这个文件用于说明：当在这个仓库里工作时，应该如何快速理解项目、数据、脚本入口和当前研究主线。

如果想先看整仓库的 Markdown / README 索引，直接看：

- `docs/repo_index.md`

## 仓库当前状态

这是一个围绕**面部运动分解**展开的研究项目，核心对象是：

- 面部关键点距离矩阵
- 距离矩阵差分
- 可解释的共享运动基
- 面瘫分级相关的侧别、严重度与数据集差异

当前主线已经不再是“同时平行探索很多候选分解法”，而是收敛为三层：

1. `scripts/lq`
   - 负责前置结构探索与消融。
2. `scripts/disentangleNet`
   - 负责当前接受版 `v31` 的冻结训练与分析闭包。
3. `scripts/matrix_vis`
   - 负责把 basis / 距离差观测解释成可视化的局部运动轨迹。

## 研究核心问题

当前核心问题可以概括成：

> 能否把窗口级面部距离差矩阵分解成跨被试共享、可解释的子运动基，同时尽量把侧别信息路由到显式 side branch，并用后验可视化验证这些 basis 的语义。

关键技术直觉：

- 直接用原始距离矩阵时，静态结构和身份信息会很强。
- 使用**距离矩阵差分**：
  - `ΔD = D_t - D_{t-1}`
  - 更容易突出运动变化而不是身份静态结构。

## 方法演化概览

截至目前，可以把方法演化粗略理解成：

1. `NMF`
   - 失败。
   - 因为 `ΔD` 含负值，而标准 NMF 依赖非负输入。
2. `SVD`
   - 成功证明 `ΔD + SVD` 可以提取运动语义。
   - 典型现象是 PC1 对 mouth 区域更敏感。
3. `Grassmann`
   - 用来验证共享基是否“真共享”，而不是联合分解时被强行算出来的。
4. `Tucker`
   - 作为尝试保留，但并不是当前合适主线。
5. `DMD`
   - 更强调时间动态，尤其适合 `win5-step5` 这种窗口更多的设置。
6. 深度结构主线
   - 从 `scripts/lq` 演化到 `scripts/disentangleNet/v31`。

## 数据说明

### 数据集

| 数据集 | 路径 | 规模 | 说明 |
|--------|------|------|------|
| IMR | `data/*/IMR/` | 约 227 | 相对受控环境 |
| TT | `data/*/TT/` | 约 42 | 场景更异质 |

### 窗口配置

| 配置 | 路径 | 主要用途 |
|------|------|----------|
| `win20-step20` | `data/win20-step20/` | 早期 SVD 与当前主线训练 |
| `win10-step10` | `data/win10-step10/` | 中间配置 |
| `win5-step5` | `data/win5-step5/` | DMD 与更稳定的相关性分析 |

补充说明：

- 单个矩阵通常是 `341 × 341` 的成对距离矩阵。
- `win20-step20` 下 IMR 患者的窗口数常常只有约 5 个。
  - 这会让 Pearson 相关性很不稳定。
- `win5-step5` 通常能提供 `25-130+` 个窗口。
  - 更适合做动态相关性分析。

### Blendshape / AU 标注

- `data/blendshape/`
  - 包含 AU / blendshape 标注。
- 主要使用：
  - `data/blendshape/IMR/`
  - `data/blendshape/TT/`

### Landmark 分区

当前 341 个点采用 grouped ordering，常用区域边界是：

```python
boundaries = [18, 38, 100, 130, 140, 188, 233, 307, 331, 341]
regions = [
    "forehead", "eyebrow", "eyehole", "eye_contour",
    "eye_iris", "nose", "around_mouth", "mouth", "cheek", "jaw"
]
```

## 代码入口速览

### 早期 feasibility：`scripts/pilot_feasibility/`

#### SVD

| 脚本 | 作用 |
|------|------|
| `scripts/pilot_feasibility/svd/svd_single_patient.py` | 单患者 SVD 差分分解 |
| `scripts/pilot_feasibility/svd/svd_multi_patient.py` | 多患者联合 SVD |
| `scripts/pilot_feasibility/svd/svd_single_patient_raw.py` | RAW vs DIFF 对比 |
| `scripts/pilot_feasibility/svd/svd_multi_patient_win5.py` | `win5-step5` 上的 SVD |

#### 其他分解法

| 脚本 | 作用 |
|------|------|
| `scripts/pilot_feasibility/nmf/nmf_baseline_x_y.py` | NMF baseline 尝试 |
| `scripts/pilot_feasibility/tucker/tucker_multi_patient.py` | Tucker 尝试 |

#### DMD

| 脚本 | 作用 |
|------|------|
| `scripts/pilot_feasibility/dmd/dmd_single_patient.py` | 单患者 DMD |
| `scripts/pilot_feasibility/dmd/dmd_multi_patient.py` | 多患者联合 DMD |
| `scripts/pilot_feasibility/dmd/dmd_blendshape_correlation.py` | DMD 与 blendshape 相关性 |

#### Blendshape / Grassmann

| 脚本 | 作用 |
|------|------|
| `scripts/pilot_feasibility/blendshape/blendshape_correlation_analysis.py` | `win20` SVD 时间系数与 blendshape 相关性 |
| `scripts/pilot_feasibility/blendshape/blendshape_correlation_analysis_win5.py` | `win5` 版本 |
| `scripts/pilot_feasibility/blendshape/blendshape_correlation_analysis_win10.py` | `win10` 版本 |
| `scripts/pilot_feasibility/grassmann/grassmann_analysis.py` | 早期 Grassmann 分析 |
| `scripts/pilot_feasibility/grassmann/grassmann_cross_analysis.py` | 跨数据集 Grassmann 验证 |

### 当前主线

- `scripts/lq/`
  - 前置结构探索。
- `scripts/disentangleNet/`
  - 当前冻结主线 `v31`。
- `scripts/matrix_vis/`
  - 后验轨迹重建和可视化。
- `scripts/val_codebook/`
  - 单患者 SVD PC1 的公共码本验证实验。

## 环境

默认环境：

```bash
conda activate openmmlab
```

## 相关外部项目

### `corelation-lm`

路径：

- `/home/weizilin/code_reproduction/corelation-lm/`

主要与 landmark ordering、热图渲染、区域配置和距离度量相关。

### `corelation-classify`

路径：

- `/home/weizilin/code_reproduction/corelation-classsify/`

主要与关键点相似性矩阵分类分析和早期差分热图探索相关。

## 当前关键实验结论

### 1. `SVD + ΔD` 能提取运动语义

- PC1 的 dominant region 在大量结果里都偏向 `mouth`。
- 如果直接用 RAW 距离矩阵，PC1 往往更像静态结构或身份信息。
- 这说明：
  - `ΔD` 对当前问题是必要的。

### 2. 多患者联合 SVD 仍然能保留运动语义

典型现象：

| 数据集 | X-PC1 | X-PC2 | PC1 dominant |
|--------|------|------|--------------|
| IMR | 64.6% | 25.1% | mouth |
| TT | 47.0% | 27.2% | mouth |

### 3. Grassmann 验证支持“共享基是真共享”

典型结果：

| 比较 | X-PC1 | Y-PC1 |
|------|------|------|
| IMR single vs IMR joint | 12.9° | 10.0° |
| TT single vs TT joint | 14.8° | 8.9° |
| TT joint vs IMR joint | 13.5° | 7.1° |

结论：

- 患者与本数据集联合基更接近。
- 说明共享基不是纯计算伪影。

### 4. `DMD` 在更密窗口上更适合相关性分析

在 `win5-step5` 上：

- 窗口更多
- 时间动态更清晰
- 与 blendshape 的相关性分析更稳

### 5. 深度主线已经收敛到 `disentangleNet/v31`

当前最重要的研究问题已经不是：

- “哪种传统分解法更好”

而是：

1. side 信息是否被稳定压入 side branch
2. free / private 里还残留多少 dataset leakage
3. learned basis 是否能通过 `matrix_vis` 形成可信的后验解释

## 当前建议先读哪些文档

- `IDEA_REPORT.md`
  - 当前主线的高层判断。
- `IDEA_EXPERIMENTS.md`
  - 最详细的实验账本。
- `RESEARCH_PROGRESS.md`
  - 当前结论、风险与下一步。
- `docs/repo_index.md`
  - 仓库 Markdown / README 总索引。

## 备注

- 当前仓库里的很多结论已经从“多方法发散探索”转向“围绕 `lq -> disentangleNet -> matrix_vis` 主线收束”。
- 阅读时不要再把 `scripts/lq` 和 `scripts/disentangleNet` 视为同一阶段。
