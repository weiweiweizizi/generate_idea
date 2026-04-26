# Matrix Vis Basis Motion Reconstruction Design

**状态**: 已确认

**日期**: 2026-04-26

---

## 1. 背景与目标

`scripts/disentangleNet` 当前已经能稳定产出可分析的 shared basis / side basis，但后验分析仍停留在矩阵热图、probe 和统计指标层面。下一步需要把这些 basis 从“距离变化矩阵”还原成更直观的 face mesh 运动，回答两个更接近语义解释的问题：

1. 一个 basis 在单轴上更像什么动作趋势。
2. `x` / `y` 两个方向组合后，对应的局部面部运动轨迹是什么样。

本设计的目标不是一次性做完所有真实数据适配，而是先在 `scripts/matrix_vis` 下搭一个独立、可扩展、可复算的后验分析框架，使后续可以逐步接入：

- 标准 face mesh 模板
- 单方向 basis 矩阵
- 固定点与点子集配置
- `disentangleNet` 导出的 shared / side basis

核心建模约束采用 [`scripts/matrix_vis/modeling.md`](/home/weizilin/generate_idea/scripts/matrix_vis/modeling.md) 中给出的凸二次规划思路：用 pairwise 平均距离变化约束、平滑正则和不可交叉约束，恢复单轴轨迹。

---

## 2. 设计目标

### 2.1 第一阶段必须做到

- 在 `scripts/matrix_vis` 下建立独立目录和 CLI 入口。
- 使用 `yaml` 作为标准配置入口，CLI 只做少量覆盖。
- 一次只重建一个方向 `x` 或 `y`。
- 输入标准 mesh、点子集、anchor 点和单方向 basis 方阵。
- 用 `cvxpy + osqp` 装配并求解 QP。
- 导出单方向结果、诊断文件和基础静态可视化。
- 支持后续把独立求好的 `x` / `y` 结果合成为 2D 运动并渲染。

### 2.2 第一阶段明确不做

- 不自动从 `disentangleNet` checkpoint 直接抽 basis。
- 不把子集运动传播到全脸所有点。
- 不做多个 basis 的联合时序求解。
- 不做交互式前端。
- 不承诺自动给 basis 命名，只负责轨迹重建和可视化。

---

## 3. 总体方案

采用“CLI 主导、内部模块化”的结构。

### 3.1 外层命令

- `reconstruct_axis.py`
  - 单方向轨迹重建主入口。
- `compose_motion.py`
  - 读取已求好的 `x` / `y` 方向结果，合成 2D 轨迹并渲染。
- `inspect_config.py`
  - 只做输入检查和 dry-run，不求解。

### 3.2 内层模块边界

1. `io/`
   - 配置读取
   - mesh / basis 输入
   - 结果落盘
2. `core/`
   - 领域对象定义
   - mesh 投影
   - 点子集映射
   - basis 观测表生成
3. `qp/`
   - 变量索引
   - 数据项与正则项装配
   - 约束装配
   - `cvxpy` + `osqp` 求解
4. `viz/`
   - 单方向静态图
   - 双方向合成预览
   - gif / 帧图导出

---

## 4. 目录结构

建议目录如下：

```text
scripts/matrix_vis/
  README.md
  modeling.md
  configs/
    examples/
      axis_x_demo.yaml
      axis_y_demo.yaml
      compose_demo.yaml
  cli/
    reconstruct_axis.py
    compose_motion.py
    inspect_config.py
  core/
    types.py
    mesh.py
    projection.py
    basis.py
    observations.py
  io/
    config.py
    load_mesh.py
    load_basis.py
    save_results.py
  qp/
    variables.py
    objective.py
    constraints.py
    builder.py
    solve.py
  viz/
    axis_plots.py
    mesh_animation.py
    exporters.py
  tests/
    test_config.py
    test_projection.py
    test_observations.py
    test_qp_builder.py
    test_compose_motion.py
```

说明：

- `cli/` 只负责编排。
- `core/` 和 `qp/` 负责数学与数据契约。
- `viz/` 不能反向依赖 `cli/`。
- `tests/` 先覆盖配置、投影、观测表和最小 QP 装配，不要求第一版就有大规模数值正确性基准。

---

## 5. 关键数据对象

为了避免脚本之间直接传裸数组，建议定义以下稳定对象：

- `MeshTemplate`
  - 全量点坐标
  - 点 ID
  - 维度信息（2D 或 3D）
- `AxisProjection`
  - 指定方向上的初始 1D 坐标
  - 子集点 ID
  - 子集坐标
- `BasisObservation`
  - 子集点 ID
  - basis 方阵
  - 观测语义标记（平均 pairwise 距离变化）
- `QPConfig`
  - 时间步数
  - 正则权重
  - 是否强制保持顺序
  - 位移边界
- `TrajectorySolution`
  - 时间网格
  - 子集点轨迹
  - 初始位置
  - anchor 点
  - solver 诊断
- `ComposedMotion`
  - 合成后的 2D 轨迹
  - 参与合成的点集
  - 静态 mesh 基底

这些对象都应尽量是轻量 dataclass，便于调试和序列化。

---

## 6. 配置格式

标准入口采用 `yaml`，建议结构如下：

```yaml
experiment:
  name: demo_x_basis0
  output_dir: outputs/matrix_vis/demo_x_basis0

mesh:
  source: path/to/standard_face_mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto

projection:
  axis: x
  source_axis_index: 0
  subset_point_ids: [188, 189, 190, 191]
  anchor_point_id: 188

basis:
  source: path/to/basis.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta

solver:
  num_time_steps: 25
  lambda_data: 1.0
  lambda_acc: 10.0
  lambda_vel: 1.0
  enforce_order: true
  max_displacement: null
  qp_backend: osqp

export:
  save_projected_mesh: true
  save_qp_diagnostics: true
  save_axis_plot: true
  save_npz: true
  save_json_summary: true
```

关键约束：

- `subset_point_ids` 必须使用全局点 ID。
- `anchor_point_id` 必须出现在 `subset_point_ids` 中。
- basis 方阵边长必须等于 `subset_point_ids` 长度。
- 第一阶段只支持单个 basis 方阵或可索引的 basis 堆叠。

---

## 7. 单方向重建流程

`reconstruct_axis.py` 的标准流程固定为：

1. 读取 `yaml` 配置并建立输出目录。
2. 加载标准 mesh。
3. 按配置投影到 `x` 或 `y` 轴，得到初始 1D 坐标。
4. 提取 `subset_point_ids` 对应子集。
5. 加载单方向 basis 方阵并校验形状。
6. 生成 pairwise 观测表。
7. 装配 QP：
   - 数据项
   - 加速度平滑项
   - 速度变化平滑项
   - 初始条件约束
   - anchor 固定点约束
   - 顺序不交叉约束
   - 可选位移边界约束
8. 用 `cvxpy + osqp` 求解。
9. 导出标准结果文件和基础图形。

### 7.1 单方向标准输出

每次运行至少输出：

- `resolved_config.yaml`
- `projected_mesh.csv`
- `observations.csv`
- `solution.npz`
- `summary.json`
- `qp_diagnostics.json`
- `axis_trajectory.png`
- `axis_displacement.png`
- `basis_fit_error.png`

---

## 8. 双方向合成流程

`compose_motion.py` 不参与求解，只消费单方向结果：

1. 读取静态标准 mesh。
2. 读取 `x` 方向单方向结果。
3. 读取 `y` 方向单方向结果。
4. 对齐两边的点 ID。
5. 合成 2D 轨迹。
6. 对未参与求解的点保持静止。
7. 渲染逐帧散点、轨迹尾迹或 gif。
8. 保存标准合成产物。

### 8.1 双方向标准输出

- `composed_motion.npz`
- `composed_summary.json`
- `motion_preview.gif` 或 `frames/*.png`
- `motion_snapshot.png`

---

## 9. 分阶段实施策略

### Phase 0：框架搭建

- 建目录与模块骨架。
- 完成配置读取、基础 dataclass、输入检查。
- 跑通 `inspect_config.py`。

### Phase 1：单方向最小闭环

- 接通 mesh 投影、basis 读取、观测表生成、QP 装配和求解。
- 用 toy 数据完成单方向闭环。

### Phase 2：单方向诊断增强

- 增加求解诊断、误差分析和静态图输出。
- 强化输入校验和错误信息。

### Phase 3：双方向合成与可视化

- 读取 `x` / `y` 结果并合成 2D 轨迹。
- 输出静态预览和 gif。

### Phase 4：对接 `disentangleNet`

- 增加 basis 适配器。
- 区分 shared basis / side basis。
- 支持批量导出基础分析任务。

---

## 10. 风险与约束

### 10.1 数值层面风险

- basis 观测可能与单轴不可交叉约束冲突，导致 QP 不可行。
- anchor 点选取不合理时，解会出现整体漂移或局部挤压。
- 子集点太少时，轨迹自由度不足；太多时，QP 规模会快速上升。

### 10.2 数据层面风险

- 后续真实 mesh 点序和 basis 点序可能不一致。
- 不同来源的 mesh 可能是 2D、3D 或不同单位尺度。
- `x` 和 `y` 两个方向使用的点子集可能不完全相同。

### 10.3 工程层面风险

- `cvxpy` / `osqp` 是新增依赖，需要提前做缺依赖提示。
- 第一阶段若同时兼容过多输入格式，会显著提高维护成本。

因此第一版坚持：

- 先收紧输入格式。
- 先解决单方向闭环。
- 先保证中间产物可检查。

---

## 11. 验收标准

第一阶段完成后，应满足以下最低验收标准：

1. 能通过一个示例 `yaml` 读取 mesh、basis、点子集和 solver 参数。
2. `inspect_config.py` 能在不求解的情况下报告输入一致性问题。
3. `reconstruct_axis.py` 能在 toy 数据上生成单方向轨迹结果与诊断文件。
4. `compose_motion.py` 能读取两个单方向结果并生成 2D 预览图。
5. 所有关键中间产物都落盘，便于逐步替换成真实 `disentangleNet` 数据。

---

## 12. 当前未决问题

以下问题有意留到真实数据接入后再定：

- 标准 mesh 的文件格式与字段约定。
- basis 文件究竟来自 checkpoint、`analysis` 产物还是单独导出的 `.npy`。
- 是否需要对子集外点做插值传播。
- 是否需要对不同 pair 施加非均匀权重。
- 是否需要支持批量 basis sweep。

本设计的重点是先把系统边界、流程顺序和文件契约固定下来，为后续逐步接实数据留出稳定接口。
