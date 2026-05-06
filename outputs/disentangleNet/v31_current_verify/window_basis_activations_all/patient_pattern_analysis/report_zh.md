# DisentangleNet 患者级激活模式与共激活模式中文汇总

本报告汇总以下两类结果：

- 患者级画像汇总：`patient_profile_summary/`
- 按类别分组的共激活分析：`coactivation_by_<class_col>/<feature_family>/`

数据来源为窗口级 basis 导出结果：

- `outputs/disentangleNet/v31_current_verify/window_basis_activations_all/window_basis_activations_wide.csv`

## 1. 数据范围与解释口径

- 患者数：`267`
- 平均每位患者窗口数：`1.685`
- 患者窗口数中位数：`1`
- 分析单位：患者
- 聚合方式：先对每位患者的窗口级指标取均值，再比较不同分组的患者级模式

本轮共激活分析只使用 8 个波动较明显的 basis：

- `b0, b1, b4, b5, b6, b8, b9, b10`

其中：

- `b0/b1` 来自 free level-0
- `b4/b5/b6` 来自 free level-1
- `b8/b9/b10` 来自 side branch

需要先说明一个解释边界：

- `b0/b1` 与 `b4/b5/b6` 处在同层 softmax/simplex 约束下，因此这些 basis 之间会天然出现较强互补相关。
- 因而真正更有语义解释价值的证据，应优先看 `b8/b9/b10` 之间，以及它们和 free branch 之间的 cross-branch 共激活边。

## 2. 患者级激活模式主结论

### 2.1 患者差异主要集中在 side branch

按患者级标准差排序：

- usage 变化最大的是 `b6`, `b5`, `b10`, `b9`, `b8`
- activation 变化最大的是 `b10`, `b9`, `b0`, `b8`

最关键的现象是：

- `free_b0` 在全部 `267` 位患者中几乎固定为 level-0 主导 basis，free level-0 基本不提供患者分型能力。
- free branch 剩余的患者差异主要来自 level-1 的 `b5` 和 `b6`。
- 真正最强的患者间差异来自 side branch，尤其是 `b10` 和 `b9`。

### 2.2 患者可分为三种稳定模式

按 `side_label_name` 汇总，患者模式非常清楚：

- `Left`
  - 76 位患者
  - 63 位落入 `left_like_b10_pure_positive`
  - `b10` 几乎纯占优，`side_coeff` 明显为正，`side_entropy` 很低
- `Normal`
  - 97 位患者
  - 87 位落入 `normal_like_b9_b10_negative`
  - 以 `b9 + b10` 混合为主，但整体 `side_coeff` 为负，`b9/b10` activation 偏负
- `Right`
  - 94 位患者
  - 86 位落入 `right_like_diffuse_low_coeff`
  - `b8/b9/b10` 更分散，`side_entropy` 最高，`side_coeff` 接近 0

这说明 side branch 并不是简单地给出一个单 basis 标记，而是在不同类别下形成了不同的“协同激活状态”。

### 2.3 仍能看到 dataset leakage

按 `dataset_name` 汇总：

- `IMR`: `225` 位患者
- `TT`: `42` 位患者

稳定差异为：

- IMR 更偏 `b5`
- TT 更偏 `b6`
- IMR 的 `side_coeff` 平均更偏正
- TT 的 `side_coeff` 平均更偏负

这说明 dataset 偏差不仅存在于均值层，也存在于患者级激活模式中。

## 3. 共激活模式的总体检验

所有全局检验均基于患者级向量进行 permutation MANOVA。结果如下：

| 分组 | 患者数分布 | activation pseudo-F / p | usage pseudo-F / p | 结论 |
|:--|:--|:--|:--|:--|
| `side_label_name` | Left `76`, Normal `97`, Right `94` | `513.36 / 0.00025` | `73.08 / 0.00025` | 类别差异极强 |
| `dataset_name` | IMR `225`, TT `42` | `12.13 / 0.00125` | `90.92 / 0.00025` | 存在数据集效应 |
| `score` | 0:`97`, 1:`70`, 2:`100` | `105.70 / 0.00025` | `10.58 / 0.00025` | 严重度相关差异明确 |
| `label_5class` | 0:`38`, 1:`38`, 2:`97`, 3:`32`, 4:`62` | `343.01 / 0.00025` | `36.53 / 0.00025` | 五分类差异最强 |

结论可以直接写成：

- 不同类别之间确实存在稳定的患者级共激活模式差异。
- 最强的分层来源是 `side_label_name` 和 `label_5class`。
- `dataset_name` 也显著，但更像偏移项，而不是最主要的语义轴。

## 4. 最值得解释的共激活边

以下优先保留 side-related 与 cross-branch 边，弱化同层 softmax 造成的结构性相关。

### 4.1 按 `side_label_name`

这是最干净、也最容易解释的一组结果。

最关键的 activation 边：

- `Left vs Normal`
  - `b8-b9`: `0.830 -> -0.299`, `delta = 1.129`, `q = 0.000368`
  - `b0-b10`: `0.545 -> -0.262`, `delta = 0.807`, `q = 0.000368`
  - `b9-b10`: `0.295 -> 0.887`, `delta = -0.591`, `q = 0.000368`
- `Normal vs Right`
  - `b8-b9`: `-0.299 -> 0.729`, `delta = -1.028`, `q = 0.000500`
  - `b0-b9`: `-0.342 -> 0.523`, `delta = -0.865`, `q = 0.000500`
  - `b9-b10`: `0.887 -> 0.314`, `delta = 0.573`, `q = 0.000500`

解释上可以概括为：

- `Left` 更像 `b10` 主导且与其它分量同向联动
- `Normal` 更像 `b9-b10` 的强耦合负向状态
- `Right` 的 side branch 更 diffuse，`b8-b9` 转为正相关

### 4.2 按 `dataset_name`

数据集差异存在，但语义强度明显弱于 `side_label_name`。

最关键的 activation / usage 边：

- activation
  - `b9-b10`: IMR `0.651`, TT `0.937`, `q = 0.006998`
  - `b4-b10`: IMR `0.162`, TT `-0.295`, `q = 0.022744`
  - `b6-b9`: IMR `0.159`, TT `-0.251`, `q = 0.022744`
- usage
  - `b8-b10`: IMR `-0.805`, TT `-0.417`, `q = 0.006998`

解释上更适合写成：

- 两个数据集在 `b9-b10` 的耦合强度、以及 side branch 与 free level-1 的联动方向上不完全一致。
- 这提示模型仍保留一定 dataset-specific 编码。

### 4.3 按 `score`

`score` 的 activation 差异非常强，usage 也显著。

最值得强调的 activation 边：

- `0 vs 1`
  - `b8-b9`: `-0.299 -> 0.808`, `q = 0.000700`
  - `b9-b10`: `0.887 -> 0.292`, `q = 0.000700`
- `0 vs 2`
  - `b8-b9`: `-0.299 -> 0.644`, `q = 0.000437`
  - `b9-b10`: `0.887 -> 0.088`, `q = 0.000437`
  - `b0-b9`: `-0.342 -> 0.333`, `q = 0.000437`

usage 版也支持同样趋势：

- `0 vs 2`
  - `b8-b9`: `-0.085 -> 0.969`, `q = 0.000875`
- `0 vs 1`
  - `b8-b9`: `-0.085 -> 0.843`, `q = 0.001750`

解释上可以写成：

- 从 `score=0` 到 `score=1/2`，`b8-b9` 从负相关切换为强正相关，是最稳定的一条严重度主线。
- 同时 `b9-b10` 耦合逐步减弱，说明严重度上升时 side branch 的协同结构发生了重组。

### 4.4 按 `label_5class`

这是全套分析里最强的一组结果。

先看总体：

- activation 显著边数：`97`
- usage 显著边数：`151`

最值得保留的 activation 边：

- `0 vs 2`
  - `b8-b9`: `0.893 -> -0.299`, `q = 0.000700`
  - `b9-b10`: `0.051 -> 0.887`, `q = 0.000700`
  - `b0-b10`: `0.509 -> -0.262`, `q = 0.000700`
- `2 vs 4`
  - `b9-b10`: `0.887 -> -0.364`, `q = 0.000437`
  - `b8-b9`: `-0.299 -> 0.628`, `q = 0.000437`
  - `b0-b9`: `-0.342 -> 0.567`, `q = 0.000437`
- `1 vs 4`
  - `b0-b10`: `0.572 -> -0.821`, `q = 0.002333`
  - `b9-b10`: `0.370 -> -0.364`, `q = 0.002333`

其中一个重要现象是：

- `label_5class` 的 `0` 和 `1` 彼此之间没有 FDR 显著差异
- 但它们和 `2/3/4` 的差异很强

因此更合理的表述是：

- 五分类并不是均匀分开的，而更像存在若干“共激活状态簇”
- 其中 `0/1` 更接近，`2` 是一个强过渡态，`4` 与前面几类在 `b9-b10` 和 `b0-b10` 上出现明显反转

## 5. 如何使用这些结果来解释 basis 语义

如果目的是给 basis 赋予更具体的语义，本轮结果支持以下工作假设：

### 假设 1

- `b10` 是最强的 side 主导 basis
- 它在 `Left` 中表现为高正激活、低熵、接近纯 usage
- 在 `Normal` 中则参与负向的 `b9-b10` 耦合

### 假设 2

- `b9` 不是单独工作的 basis
- 它更像一个“关系型 basis”，其语义取决于与 `b8`、`b10` 以及 free branch 的共激活方向

### 假设 3

- `b8-b9` 的相关方向是一个高价值语义指标
- 它能够同时区分 `Left/Normal/Right`，也能够区分 `score` 与 `label_5class`

### 假设 4

- `b5/b6` 更像 dataset-sensitive 的 free level-1 分解轴
- 目前它们的语义稳定性弱于 side branch

## 6. 推荐优先查看的图

如果只看少量图，建议优先看下面这些：

### 患者画像

- `patient_profile_summary/report.md`

### side_label_name

- `coactivation_by_side_label_name/activation/mean_activation_by_side_label_name.png`
- `coactivation_by_side_label_name/activation/correlation_heatmaps/`
- `coactivation_by_side_label_name/activation/delta_heatmaps/`

### dataset_name

- `coactivation_by_dataset_name/activation/mean_activation_by_dataset_name.png`
- `coactivation_by_dataset_name/activation/delta_heatmaps/`

### score

- `coactivation_by_score/activation/mean_activation_by_score.png`
- `coactivation_by_score/activation/delta_heatmaps/`

### label_5class

- `coactivation_by_label_5class/activation/mean_activation_by_label_5class.png`
- `coactivation_by_label_5class/activation/delta_heatmaps/`

## 7. 当前最稳妥的总结写法

如果要在汇报或论文里先给一个保守但有说服力的结论，建议写成：

1. `disentangleNet` 的患者级 basis 激活并非随机波动，而是在 `side_label_name`、`score` 和 `label_5class` 下形成了显著不同的共激活结构。
2. 最稳定的差异集中在 side branch，尤其是 `b8-b9` 与 `b9-b10` 两组关系，以及 `b10` 与 free branch 的 cross-branch 耦合。
3. `Left`、`Normal`、`Right` 并不是简单由单 basis 区分，而是对应三种不同的 side-branch 协同激活模式。
4. `dataset_name` 也存在显著差异，提示当前表示仍保留一定 dataset leakage，因此后续在解释 basis 语义时应优先强调跨数据集稳定的 side-related 共激活边。
