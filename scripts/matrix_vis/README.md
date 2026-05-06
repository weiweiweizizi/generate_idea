# matrix_vis

`scripts/matrix_vis` 是一个面向单轴窗口差观测的后验重建工具。它的目标不是从 basis 严格反演出唯一真实轨迹，而是：

- 明确当前运动假设
- 在这些假设下重建一个可解释的窗口内轨迹
- 导出图、轨迹和诊断，便于后续判断这个解释是否可信

当前 `main` 分支的实现应被理解为一个**单阶段迭代线性化 QP sandbox**，不是两阶段结构投影器。

如果你是第一次使用这个工具，先看英文版 [USAGE_GUIDE.md](/home/weizilin/generate_idea/scripts/matrix_vis/USAGE_GUIDE.md) 或中文版 [USAGE_GUIDE_CN.md](/home/weizilin/generate_idea/scripts/matrix_vis/USAGE_GUIDE_CN.md)。

## 当前问题定义

当前背景假设是：

- 数据按固定长度窗口切分
- 对每个窗口、每个轴 (`x` / `y`) 都有一个窗口平均距离矩阵
- 当前输入是相邻窗口平均距离矩阵之差，也就是 `diff of distance matrix`
- 上一窗口被视为已知参考窗口，参考位置是 `x_i(0)`
- 目标是在下一窗口内恢复一个长度已知的单轴运动轨迹

严格写法见 [modeling.md](/home/weizilin/generate_idea/scripts/matrix_vis/modeling.md)。

## 当前实现到底在做什么

主线实现不是先恢复结构再生成轨迹，而是一步直接优化整条轨迹。

核心流程：

1. 读取 mesh、subset、anchor、单轴观测矩阵
2. 把矩阵转成点对 observation table
3. 建立逐点逐帧变量 `x_var`
4. 用迭代线性化去拟合单轴 `diff of distance matrix`
5. 用时间平滑正则挑出一个更稳定的解
6. 输出轨迹、图、比较指标和 `x/y` 合成结果

所以当前版本更准确的理解是：

> 给定单轴窗口差观测，求一个满足 anchor 约束且时间上足够平滑的轨迹解释。

## 当前观测语义

当前 toy 和当前 solver 已经对齐到下面这个观测定义：

\[
\Delta D_{ij}
=
\frac{1}{T}\sum_t |x_j(t)-x_i(t)|
- |x_j(0)-x_i(0)|
\]

这意味着：

- `distance matrix` 本身非负
- `diff of distance matrix` 可正可负

当前实现不再使用旧版本那种“平均位移差 `\bar\delta_j-\bar\delta_i`”作为主观测语义。

## 模块边界

结合最早的设计 spec，当前这套工具的边界仍然适合按下面的方式理解：

- `cli/`
  - 只负责编排，不写数学逻辑
- `pipelines/`
  - 负责单轴重建、配置检查、`x/y` 合成等流程编排
- `io/`
  - 配置读取、mesh / basis 输入、结果落盘
- `core/`
  - dataclass、mesh 投影、点子集映射、observation table
- `qp/`
  - 变量布局、数据项、正则项、约束、求解
- `viz/`
  - 单轴图、ground truth 对比图、`x/y` 合成预览
- `tests/`
  - 配置、投影、最小 QP 装配与 toy 回归

这套边界和 2026-04-26 的原始设计是一致的，只是当前数学实现已经从当时设想的两阶段方案回到了单阶段版本。

## 稳定数据对象

虽然 `matrix_vis` 不是一个打包库，但内部数据契约已经基本稳定，当前最值得记住的是这些对象：

- `MeshTemplate`
  - 全量点坐标、点 ID、维度信息
- `AxisProjection`
  - 指定轴上的初始 1D 坐标、子集点 ID、anchor 点
- `BasisObservation`
  - 子集点 ID、单轴观测矩阵、语义标记
- `QPConfig`
  - 时间步数、正则权重、位移边界、solver 后端
- `TrajectorySolution`
  - 子集轨迹、初始位置、时间网格、anchor 点、诊断
- `ComposedMotion`
  - `x/y` 合成后的 2D 轨迹与元数据

这些对象都定义在 [core/types.py](/home/weizilin/generate_idea/scripts/matrix_vis/core/types.py:1)，它们是当前 `matrix_vis` 内部最稳定的一层接口。

## 当前算法骨架

当前主线版本的算法部分在：

- [qp/builder.py](/home/weizilin/generate_idea/scripts/matrix_vis/qp/builder.py:1)
- [qp/objective.py](/home/weizilin/generate_idea/scripts/matrix_vis/qp/objective.py:1)
- [qp/constraints.py](/home/weizilin/generate_idea/scripts/matrix_vis/qp/constraints.py:1)
- [qp/solve.py](/home/weizilin/generate_idea/scripts/matrix_vis/qp/solve.py:1)

主要特征：

- 单阶段逐帧轨迹变量
- OSQP 求解
- 外层最多 4 轮迭代线性化
- 多个 anchor 点同时固定
- 二阶时间平滑
- 速度起伏正则
- 基于初始 gap 的 point-pair weighting

当前主线**没有**这些内容：

- 两阶段 Stage A / Stage C 结构
- structural residual 导出
- 全局顺序约束
- 局部邻接平滑正则

这些都讨论过或试验过，但不属于当前 `main` 的已采用实现。

## 当前正则和约束

当前保留的是一组相对弱、相对稳的约束：

- 初始帧固定
- anchor 点全程固定
- 可选 `max_displacement`

当前正则包括：

- 二阶差分平滑项
- 速度相对自身均值的偏离项

后一项的目的不是压小整体运动，而是减少速度起伏。

## 为什么 `x` 通常比 `y` 难

在当前 toy 和当前问题设定里：

- `y` 更接近“开口”这种单一主运动
- `x` 更像固定边界下的内部横向重排

所以 `x` 轴通常：

- 可识别性更弱
- 更依赖正则
- 更容易受局部线性化和不稳定 point pair 影响

这也是为什么当前实现专门加入了 point-pair weighting。

## CLI

- `python scripts/matrix_vis/cli/inspect_config.py --config <path>`
  - 配置检查
- `python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct --config <path>`
  - 单轴重建
- `python scripts/matrix_vis/cli/compose_motion.py compose --config <path>`
  - 合成 `x/y` 结果

## Toy 工作流

当前 example config 使用 toy 数据：

- [scripts/matrix_vis/configs/examples/axis_x_demo.yaml](/home/weizilin/generate_idea/scripts/matrix_vis/configs/examples/axis_x_demo.yaml:1)
- [scripts/matrix_vis/configs/examples/axis_y_demo.yaml](/home/weizilin/generate_idea/scripts/matrix_vis/configs/examples/axis_y_demo.yaml:1)
- [scripts/matrix_vis/configs/examples/compose_demo.yaml](/home/weizilin/generate_idea/scripts/matrix_vis/configs/examples/compose_demo.yaml:1)

toy 数据生成器：

```bash
python scripts/matrix_vis/scripts/generate_toy_double_crescent_data.py
```

它会写出：

- `mesh_2d.npy`
- `trajectory_2d.npy`
- `basis_open_mouth_x.npy`
- `basis_open_mouth_y.npy`

位置在：

- `data/toy/matrix_vis/leaf_to_rectangle_mouth_opening/`

## 输出

单轴重建通常会输出：

- `solution.npz`
- `summary.json`
- `qp_diagnostics.json`
- `projected_mesh.csv`
- `observations.csv`
- 单轴轨迹图
- 如果有 toy ground truth，则还会输出比较图和误差指标

`x/y` 合成通常会输出：

- `composed_motion.npz`
- `composed_summary.json`
- `motion_snapshot.png`
- `motion_preview.gif`

## 目录概览

当前主目录结构仍和最初 spec 规划基本一致：

```text
scripts/matrix_vis/
  README.md
  modeling.md
  configs/examples/
  cli/
  core/
  io/
  qp/
  viz/
  tests/
```

如果你要快速读代码，最推荐的顺序是：

1. `README.md`
2. `modeling.md`
3. `cli/reconstruct_axis.py`
4. `qp/builder.py`
5. `qp/objective.py`
6. `qp/solve.py`

## 范围边界

当前 `matrix_vis` 仍然是一个研究用 sandbox，不是成型模块。

它当前不负责：

- 直接接入 `disentangleNet` checkpoint
- 从局部 subset 推广到整脸
- 多个 basis 联合反演
- 交互式调参界面

如果后续要继续演化，建议以 [modeling.md](/home/weizilin/generate_idea/scripts/matrix_vis/modeling.md) 为准，而不要再以历史两阶段草案为准。

## 当前 `disentangleNet` 桥接范围

当前第一版桥接只支持两种输入：

1. basis-wise `x` reconstruction
   - 输入来自 `disentangleNet` 导出的 `basis_bank_x.npy`
   - 当前布局不是纯 `mouth` 42 点，而是 `face_regions_grouped` 下的
     `around_mouth + mouth` 119 点
   - 每个 basis 单独作为一个 observation matrix 进入 `matrix_vis`
   - 每个 `x` 结果再与固定的 `y` 结果合成

2. patient-wise coefficient-composed `x` reconstruction
   - 输入来自一个目标患者的逐窗口系数组合矩阵
   - 当前仅验证 `TT/844697`
   - `y` 在第一版里保持静止

当前桥接仍然不支持：

- 直接读取 checkpoint 并在 `matrix_vis` 内部完成推理
- 在 `matrix_vis` 内部对多个 basis 联合做系数组合
- 自动从 `mouth` basis 扩展到 full341 布局

桥接契约见：

- [docs/disentanglenet_matrix_vis_contract.md](/home/weizilin/generate_idea/docs/disentanglenet_matrix_vis_contract.md)
