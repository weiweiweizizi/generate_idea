# matrix_vis modeling

`matrix_vis` 目前采用的是一个两阶段逆问题框架，而不是把单个
`window diff` 直接反推成逐帧轨迹。

这份文档只记录当前阶段认可的建模方式。

---

## 1. 问题背景

设每个窗口长度固定为 `M` 帧。对每个窗口、每个轴 (`x` 或 `y`)，
我们得到一个 `window matrix`，它表示该窗口内点对距离行为的平均结果。

相邻两个窗口之间的差记为 `window diff`。

当前重建问题的输入是：

- 上一窗口的已知参考位置 `x_i(0)`
- 当前窗口对应的单轴 `window diff`

目标不是恢复唯一的真实轨迹，而是：

1. 先恢复当前窗口的结构目标
2. 再生成一个长度为 `M` 帧、单调、平滑、可解释的窗口内轨迹

这意味着重建本质上是一个优化问题，而不是闭式逆解。

---

## 2. Stage A: 结构目标恢复

Stage A 只负责把 `window diff` 投影到一个更可实现的窗口结构上。

### 2.1 已知量

- 上一窗口初始位置 `x_i(0)`
- 当前窗口差分矩阵 `ΔD`

### 2.2 上一窗口平均结构

先由上一窗口位置构造上一窗口的平均结构：

[
D_{\text{prev}}(i, j) = |x_j(0) - x_i(0)|
]

### 2.3 原始目标

[
D_{\text{raw}} = D_{\text{prev}} + \Delta D
]

这里的含义是：假定上一窗口保持不动，则当前窗口应该朝着什么
结构变化。

### 2.4 结构投影

`D_raw` 不一定严格可实现，因此需要求一个最近的结构代理 `z`。
`z` 不是最终轨迹，而是当前窗口的结构锚点。

可以把它理解为：

[
z = \arg\min_z \; \mathcal{L}_{struct}(z; D_{\text{raw}}, x(0))
]

其中 `\mathcal{L}_{struct}` 至少应该包含：

- pairwise distance fit
- 对上一窗口位置的适度参考
- 可实现性约束 / 结构残差诊断

### 2.5 Structural Residual

这一阶段必须输出结构残差，作为输入可实现性的诊断：

- `structural_residual_rmse`
- `structural_residual_heatmap`
- `pairwise consistency` 统计

它的作用是区分：

- 输入本身就不太可实现
- 后续轨迹生成不够好

当前 toy CLI 导出会把 Stage A 结果直接落盘，至少包括：

- `stage_a_d_raw.csv`
- `stage_a_d_hat.csv`
- `stage_a_structural_residual.csv`

同时 `summary.json` / `qp_diagnostics.json` 会暴露：

- `structural_residual_rmse`
- `trajectory_fit_rmse`
- `mean_alignment_rmse`
- `sign_conflict_count`

---

## 3. Stage C: 窗口内轨迹生成

Stage C 负责把 Stage A 的结构目标变成一个长度为 `M` 帧的单调轨迹。

这里采用的是**2 到 4 个全局共享单调时间基**，而不是单一包络，也不是每个点完全自由。

### 3.1 轨迹参数化

[
x_i(t_m) = x_i(0) + \sum_{k=1}^{K} a_{ik}\,\phi_k(t_m), \quad K = 2 \sim 4
]

其中：

- `phi_k(t)` 是全局共享时间基
- `a_{ik}` 是每个点自己的系数
- `x` / `y` 两个轴各自独立求解，但共享同一种时间基族

### 3.2 时间基要求

当前阶段采用方案 1，即固定解析基：

- `phi_k(0) = 0`
- `phi_k(t)` 单调不减
- `phi_k` 仅表达“推进程度”，不表达回摆
- `phi_k` 在 `x` / `y` 两轴中共用

这和“窗口内不反向”的问题假设一致。

### 3.3 推荐的基族

建议默认使用 3 个单调基：

- `early-rise`
- `linear`
- `late-rise`

如果后续需要更细的时序形状，再考虑扩展到 4 个基。

---

## 4. Stage C 的目标函数

Stage C 的主要数据项应该直接对齐窗口平均距离目标，而不是对齐单帧位置。

### 4.1 平均距离拟合

[
\mathcal{L}_{data}
=
\sum_{i<j} w_{ij}
\Big(
\frac{1}{M}\sum_{m=0}^{M-1} |x_j(t_m)-x_i(t_m)|
- D_{\hat{}}(i,j)
\Big)^2
]

其中 `D_hat` 是 Stage A 的结构投影结果。

### 4.2 平均位置对齐

可以加入一个平均位置对齐项，约束窗口平均状态接近 `z`：

[
\mathcal{L}_{mean}
=
\sum_i
\Big(
x_i(0) + \sum_{k=1}^{K} \mu_k a_{ik} - z_i
\Big)^2
]

其中：

[
\mu_k = \frac{1}{M}\sum_{m=0}^{M-1} \phi_k(t_m)
]

### 4.3 幅值正则

[
\mathcal{L}_{amp} = \sum_{i,k} a_{ik}^2
]

作用是抑制不必要的大振幅解。

### 4.4 空间平滑

不要再使用全局的点 ID 顺序约束。更合理的是在局部邻接图上做平滑：

[
\mathcal{L}_{spatial}
=
\sum_{(i,j)\in E}\sum_k (a_{ik} - a_{jk})^2
]

这里的 `E` 应该来自嘴部拓扑或局部邻接关系，而不是点 ID 排序。

### 4.5 总目标

[
\mathcal{L}
=
\lambda_d \mathcal{L}_{data}
+
\lambda_m \mathcal{L}_{mean}
+
\lambda_a \mathcal{L}_{amp}
+
\lambda_s \mathcal{L}_{spatial}
]

---

## 5. 约束条件

当前建议保留的约束只有这些：

- anchor 点固定
- 单调时间基
- 系数符号与 Stage A 的位移方向一致
- 必要时加位移边界

### 5.1 Anchor 固定

对 anchor 点，所有系数都固定为 0。

### 5.2 符号一致性

设 Stage A 的位移方向为：

[
d_i = z_i - x_i(0)
]

则：

- 若 `d_i > eps`，约束 `a_{ik} >= 0`
- 若 `d_i < -eps`，约束 `a_{ik} <= 0`
- 若 `|d_i| <= eps`，只靠正则压小

这比全局顺序约束更符合当前问题定义。

### 5.3 已废弃的约束

以下假设已明确废弃：

- `x_{i,m} <= x_{i+1,m}` 这种基于点 ID 的全局顺序约束
- 把点 ID 顺序当作几何顺序

---

## 6. 为什么分轴处理

当前问题仍然是按轴分别建模：

- `x` 轴单独重建
- `y` 轴单独重建
- 最后再 compose 成 2D 结果

这样做的原因是：

- 输入和 basis 本来就是按轴分开的
- 单轴问题更容易诊断
- 2D 合成时可以检查两个轴是否在时间相位上兼容

因此，`x` 和 `y` 两轴可以共享同一种时间基族，但系数必须独立求解。

---

## 7. 优化问题性质

Stage A 和 Stage C 都属于约束优化问题。

当前阶段更重要的是问题定义清晰，而不是坚持某个固定求解器形式。

如果后续继续使用 QP / convex solver，也应该优先保证：

- 观测语义正确
- 结构残差可解释
- 时间模型与“窗口内不反向”一致

---

## 8. 关键直觉

这套模型想表达的是：

> 先把窗口差投影到一个可实现的结构目标上，
> 再用少量共享单调时间基，生成一个符合该结构目标的窗口内运动轨迹。

它不是在假装存在唯一逆解，而是在“可实现结构”与“平滑时序”之间找一个最合理的解释。
