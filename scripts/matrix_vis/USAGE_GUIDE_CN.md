# matrix_vis 对外使用指南

`matrix_vis` 是一个后验重建工具，用来把单轴窗口差距离矩阵转成一个可解释的局部 mesh 运动轨迹。

这份文档面向仓库外部使用者，目标是让你在不通读全部源码的前提下，也能：

- 跑通工具
- 理解输入输出文件契约
- 明白模块边界
- 在自己的研究流程里稳定接入

## 1. `matrix_vis` 当前在做什么

当前 `main` 分支应理解为：

- 一个单轴（`x` 或 `y`）重建工作流
- 由 YAML 配置驱动
- 使用 mesh 模板、点子集和一个方阵观测矩阵
- 求解一个窗口内足够平滑的轨迹解释
- 可选把已经求出的 `x` / `y` 轨迹合成为 2D 动画预览

它目前还不是通用库，更准确地说，它是一个研究型 sandbox，但内部接口已经稳定到足以被脚本和可复现实验配置直接使用。

## 2. 项目结构

```text
scripts/matrix_vis/
  README.md
  modeling.md
  USAGE_GUIDE.md
  USAGE_GUIDE_CN.md
  cli/
  configs/
    examples/
    landmarks/
    real/
  core/
  io/
  pipelines/
  qp/
  scripts/
  tests/
  viz/
```

建议按下面的职责理解这些目录：

- `cli/`
  - 面向用户暴露的命令行入口
  - 应尽量保持轻量，只做参数接收并转发到 pipeline
- `pipelines/`
  - 端到端流程编排层
  - 如果你想快速理解“这个工具实际怎么跑”，这里最值得先读
- `io/`
  - 配置解析
  - mesh 加载
  - basis / 观测矩阵加载
  - 结果保存
- `core/`
  - 相对稳定的数据对象
  - 轻量级几何和数据变换
  - 包括单轴投影、方阵转 observation table 等
- `qp/`
  - 优化问题装配与求解
- `viz/`
  - 轨迹图、静态预览、逐帧导出、GIF 导出
- `configs/examples/`
  - 最小 toy 示例
- `configs/real/`
  - 仓库里实际使用的真实实验配置
- `tests/`
  - 配置解析、观测加载、投影、QP 装配等回归测试

## 3. 主要用户入口

### 3.1 检查单轴配置

```bash
python scripts/matrix_vis/cli/inspect_config.py inspect --config scripts/matrix_vis/configs/examples/axis_x_demo.yaml
```

它会做的事情：

- 加载 YAML 配置
- 解析 mesh 和 basis 输入
- 计算投影后的 subset
- 检查 subset 和 basis 方阵之间的形状契约
- 向 stdout 打印一个 summary JSON

建议你在新配置第一次正式求解之前，先跑一次 `inspect`。

### 3.2 重建单个轴

```bash
python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct --config scripts/matrix_vis/configs/examples/axis_x_demo.yaml
python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct --config scripts/matrix_vis/configs/examples/axis_y_demo.yaml
```

可选输出目录覆盖：

```bash
python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct \
  --config path/to/config.yaml \
  --output_dir outputs/matrix_vis/custom_run
```

它会做的事情：

- 加载配置
- 加载 mesh
- 把 mesh 投影到 `x` 或 `y`
- 加载 basis 矩阵，或者加载 `next - prev` 的矩阵差
- 把方阵转为 pairwise observations
- 组装 QP bundle
- 求解轨迹
- 导出图、summary 和 `solution.npz`

### 3.3 合成已求解的 `x` / `y` 结果

```bash
python scripts/matrix_vis/cli/compose_motion.py compose --config scripts/matrix_vis/configs/examples/compose_demo.yaml
```

它会做的事情：

- 加载一个 mesh
- 加载 `x` 轴重建输出的 `solution.npz`
- 加载 `y` 轴重建输出的 `solution.npz`
- 取二者点集交集
- 把重建得到的 `x/y` 坐标按时间写回 mesh
- 导出静态快照、可选 GIF 和 `composed_motion.npz`

## 4. 端到端 Dataflow

这是当前版本实际使用的数据流。

### 4.1 单轴重建 dataflow

```text
axis config yaml
  -> io.config.load_config
  -> io.load_mesh.load_mesh
  -> core.projection.project_mesh_to_axis
  -> io.load_basis.load_basis_observation
  -> core.observations.basis_to_observation_table
  -> qp.builder.build_axis_qp
  -> qp.solve.solve_axis_qp
  -> io.save_results + viz.axis_plots
  -> summary.json / solution.npz / diagnostics / plots
```

### 4.2 `x/y` 合成 dataflow

```text
compose config yaml
  -> io.compose_config.load_compose_config
  -> io.load_mesh.load_mesh
  -> io.save_results.load_solution_npz (x)
  -> io.save_results.load_solution_npz (y)
  -> pipelines.compose.run_motion_composition
  -> viz.mesh_animation
  -> composed_summary.json / composed_motion.npz / preview images
```

## 5. 核心概念与契约

外部使用时，建议把下面几个概念明确分开。

### 5.1 Mesh

mesh 是求解器获取初始位置的几何模板。

当前支持的输入格式：

- `numpy`
  - 通常是 `.npy`
  - 预期形状为 `[N, D]`
- `mediapipe_canonical_obj`
  - canonical face mesh OBJ
  - loader 会读取前 468 个顶点
  - 也可以合成 iris 点扩展到 478 个点

当前支持的维度：

- `2d`
- `3d`

### 5.2 Projection

重建始终按单轴进行。

投影步骤会把 mesh 变成：

- `full_axis_positions`
  - 该轴上所有点的位置
- `subset_point_ids`
  - 当前逆问题真正参与的点 ID
- `subset_positions`
  - 当前 subset 在该轴上的初始 1D 位置
- `anchor_point_ids`
  - 在整个轨迹期间保持固定的点

### 5.3 Basis observation

当前观测是一个方阵，其语义为：

\[
\Delta D_{ij}
=
\frac{1}{T}\sum_t |x_j(t)-x_i(t)| - |x_j(0)-x_i(0)|
\]

当前支持的语义：

- `mean_distance_delta`

当前工具支持两种提供方式：

- `basis.source`
  - 直接提供 `.npy` 方阵或方阵堆栈
- `basis.prev_source` + `basis.next_source`
  - loader 会自动计算 `next - prev`

### 5.3.1 `disentangleNet` 桥接的第一版约束

当前桥接约定不是让 `matrix_vis` 直接处理 `disentangleNet` 的 latent，而是
让 `disentangleNet` 先导出可消费的 observation matrix：

- Step 1
  - 导出 `x/mouth` basis stack
  - 当前 basis 对应的是 `around_mouth + mouth` 的 119 点 grouped layout
  - 每个 basis 单独做一次 `x` 重建
  - 每个 `x` 结果再与固定 `y` 结果合成
- Step 2
  - 对目标患者先按窗口系数组合出一个 `x` observation matrix
  - 再把这个矩阵送入 `matrix_vis`
  - 第一版 `y` 保持静止

桥接契约的详细字段定义见：

- [docs/disentanglenet_matrix_vis_contract.md](/home/weizilin/generate_idea/docs/disentanglenet_matrix_vis_contract.md)

### 5.4 Observation table

方阵会被转换成上三角 pairwise 表格：

- `i`
  - subset 内的局部索引
- `j`
  - subset 内的局部索引
- `point_id_i`
  - 全局点 ID
- `point_id_j`
  - 全局点 ID
- `value`
  - 当前矩阵条目值

这个表就是 QP 组装的直接输入。

## 6. 相对稳定的 Python 接口

如果你想从自己的 Python 代码里调用 `matrix_vis`，下面这些接口最值得依赖。

### 6.1 配置 dataclass

定义在 [core/types.py](/home/weizilin/generate_idea/scripts/matrix_vis/core/types.py:21)：

- `ExperimentConfig`
- `MeshConfig`
- `ProjectionConfig`
- `BasisConfig`
- `QPConfig`
- `ExportConfig`
- `MatrixVisConfig`
- `ComposeInputConfig`
- `ComposeExportConfig`
- `ComposeConfig`

这层可以看成当前最稳定的外部契约。

### 6.2 已加载数据 dataclass

同样定义在 [core/types.py](/home/weizilin/generate_idea/scripts/matrix_vis/core/types.py:123)：

- `MeshTemplate`
- `AxisProjection`
- `BasisObservation`
- `TrajectorySolution`
- `ComposedMotion`

### 6.3 Pipeline 函数

当前最适合从 Python 侧直接调用的入口是：

- `scripts.matrix_vis.pipelines.inspect.inspect_axis_config`
- `scripts.matrix_vis.pipelines.reconstruct.run_axis_reconstruction`
- `scripts.matrix_vis.pipelines.compose.run_motion_composition`

相比直接在 Python 里调用 CLI 模块，这三个函数更合适。

## 7. 单轴配置 Schema

单轴配置由 [io/config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/config.py:165) 解析。

最小结构如下：

```yaml
experiment:
  name: demo_axis_x
  output_dir: outputs/matrix_vis/demo_axis_x

mesh:
  source: path/to/mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto

projection:
  axis: x
  source_axis_index: 0
  subset_point_ids: [0, 1, 2, 3]
  anchor_point_ids: [0, 3]

basis:
  source: path/to/basis_x.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta

solver:
  num_time_steps: 25
  lambda_data: 1.0
  lambda_acc: 10.0
  lambda_vel: 1.0
  enforce_order: false
  max_displacement: null
  qp_backend: osqp

export:
  save_projected_mesh: true
  save_qp_diagnostics: true
  save_axis_plot: true
  save_npz: true
  save_json_summary: true
```

### 7.1 `experiment`

- `name: str`
  - summary 中展示的实验名
- `output_dir: str`
  - 本次运行的输出目录

### 7.2 `mesh`

- `source: str`
  - mesh 输入路径
- `format: str`
  - `numpy` 或 `mediapipe_canonical_obj`
- `dimension: str`
  - `2d` 或 `3d`
- `point_ids: auto | [int, ...]`
  - `auto` 表示自动赋值 `0..N-1`
  - 显式列表时，长度必须与点数一致
- `normalization_scope: str | null`
  - 可选
  - 当前支持 `mouth_only`、`eye_only`、`face_regions`

### 7.3 `projection`

- `axis: str`
  - `x` 或 `y`
- `source_axis_index: int`
  - 通常 `x=0`，`y=1`
- `subset_point_ids: [int, ...]`
  - 当前要重建的点子集
- `anchor_point_ids: [int, ...]`
  - 一个或多个固定点

也支持通过 layout 指定 subset：

```yaml
projection:
  axis: x
  source_axis_index: 0
  subset_layout:
    name: face_regions_grouped
    source: scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml
    extractor_name: mediapipe
    region_names: [around_mouth, mouth]
  anchor_point_ids: [18]
```

当前支持的 `subset_layout.name`：

- `face_regions_grouped`
- `mouth`

验证规则：

- subset IDs 不能重复
- anchor IDs 不能重复
- 每个 anchor ID 都必须包含在 subset 中
- `source_axis_index` 必须与 mesh 维度兼容

### 7.4 `basis`

直接矩阵模式：

```yaml
basis:
  source: path/to/basis_x.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta
```

差分模式：

```yaml
basis:
  prev_source: path/to/window_prev.npy
  next_source: path/to/window_next.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta
```

可选的 full-matrix crop 模式：

```yaml
basis:
  prev_source: path/to/window_prev.npy
  next_source: path/to/window_next.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta
  matrix_layout:
    name: face_regions_grouped
    source: scripts/matrix_vis/configs/landmarks/mediapipe_face_regions_full.yaml
    extractor_name: mediapipe
```

什么时候需要 `matrix_layout`：

- 你加载的矩阵覆盖了一个更大的 canonical 点排序
- 但当前实验只重建其中一个子集
- 这时需要 `matrix_vis` 按点 ID 一致地裁剪方阵

### 7.5 `solver`

- `num_time_steps: int`
  - 必须 `>= 2`
- `lambda_data: float`
  - 数据项权重
- `lambda_acc: float`
  - 二阶差分时间平滑项权重
- `lambda_vel: float`
  - 速度起伏正则权重
- `enforce_order: bool`
  - 当前会被解析，但不是主要控制入口
- `max_displacement: float | null`
  - 相对初始位置的可选硬边界
- `qp_backend: str`
  - 当前支持 `osqp`、`matrix_free_cg`
- `max_observations: int | null`
  - 可选，只保留绝对值最大的若干 pairwise observations

### 7.6 `export`

- `save_projected_mesh: bool`
- `save_qp_diagnostics: bool`
- `save_axis_plot: bool`
- `save_npz: bool`
- `save_json_summary: bool`

## 8. Compose 配置 Schema

compose 配置由 [io/compose_config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/compose_config.py:45) 解析。

当前支持的结构如下：

```yaml
experiment:
  name: demo_compose_xy
  output_dir: outputs/matrix_vis/demo_compose_xy

mesh:
  source: path/to/mesh.npy
  format: numpy
  dimension: 2d
  point_ids: auto

inputs:
  x_solution: outputs/matrix_vis/demo_axis_x/solution.npz
  y_solution: outputs/matrix_vis/demo_axis_y/solution.npz

compose:
  subset_policy: intersection

export:
  save_animation_preview: true
  save_npz: true
  save_json_summary: true
```

当前重要限制：

- 只支持 `subset_policy: intersection`

这意味着：

- 合成结果只使用同时出现在 `x` 与 `y` 解中的点
- 其他 mesh 点保持静止，因为 pipeline 会先从原始 mesh 初始化全部坐标，只覆盖交集 subset

## 9. 输出文件契约

### 9.1 单轴重建输出

典型输出文件：

- `resolved_config.yaml`
- `projected_mesh.csv`
- `observations.csv`
- `solution.npz`
- `summary.json`
- `qp_diagnostics.json`
- `axis_trajectory.png`
- 当 toy ground truth 存在时，还会有 `axis_ground_truth_comparison.png`

#### `projected_mesh.csv`

列：

- `point_id`
- `axis_position`

#### `observations.csv`

列：

- `i`
- `j`
- `point_id_i`
- `point_id_j`
- `value`

#### `solution.npz`

key：

- `point_ids`
- `time_grid`
- `initial_positions`
- `trajectory`
- `anchor_point_ids`
- `anchor_point_id`
- `basis_matrix`

形状：

- `point_ids`: `[N]`
- `time_grid`: `[T]`
- `initial_positions`: `[N]`
- `trajectory`: `[N, T]`
- `anchor_point_ids`: `[K]`
- `basis_matrix`: `[N, N]`

#### `summary.json`

典型字段：

- experiment name
- output directory
- axis
- subset point count
- pairwise observation count
- truncation info
- anchor IDs
- plot warnings
- comparison metrics
- solver diagnostics

### 9.2 合成输出

典型输出文件：

- `motion_snapshot.png`
- `frames/`
- `motion_preview.gif`
- `composed_motion.npz`
- `composed_summary.json`

#### `composed_motion.npz`

key：

- `point_ids`
- `time_grid`
- `coordinates`
- `subset_point_ids`

形状：

- `point_ids`: `[N]`
- `time_grid`: `[T]`
- `coordinates`: `[T, N, D]`
- `subset_point_ids`: `[M]`

## 10. 推荐使用顺序

面对一个新数据集或新实验，建议按下面顺序走：

1. 准备一个点 ID 稳定的 mesh。
2. 决定先重建哪个轴。
3. 决定 subset 和 anchors。
4. 确认 basis 矩阵排序与 subset 排序一致。
5. 先跑 `inspect`。
6. 跑单轴 solve。
7. 检查 `summary.json`、`qp_diagnostics.json` 和轨迹图。
8. 如果需要，再跑第二个轴。
9. 运行 `compose`。
10. 检查 `motion_snapshot.png` 和 `motion_preview.gif`。

## 11. 常见失败模式

### 11.1 Basis 形状和 subset 点数不一致

常见原因：

- subset IDs 与矩阵排序不一致
- 你加载的是更大区域的 full matrix，但配置的是更小 subset

修复方式：

- 直接把 basis 矩阵预先裁到目标 subset
- 或者配置 `basis.matrix_layout`，让 loader 按点 ID 自动裁剪

### 11.2 Anchor 点不在 subset 中

常见原因：

- 配置写错
- subset 改了，但 anchor 没同步改

修复方式：

- 确认所有 `anchor_point_id` 都包含在 `subset_point_ids` 中

### 11.3 `source_axis_index` 错误

常见原因：

- `x/y` 标签与轴索引不一致
- 在 2D mesh 上误用了 `2`

修复方式：

- `x` 使用 `0`
- `y` 使用 `1`
- 只有 `3d` mesh 才能使用 `2`

### 11.4 合成时 `x` / `y` 结果没有交集

常见原因：

- 两次单轴重建用了不同 subset
- `solution.npz` 来自不兼容的配置

修复方式：

- 确保你希望合成的点集同时出现在两个轴的重建结果里

### 11.5 示例配置里还有当前实现不支持的旧字段

如果你是从旧版配置或历史实验里复制过来的，请以这两个文件为准重新核对：

- [io/config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/config.py:165)
- [io/compose_config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/compose_config.py:45)

这两个文件是当前配置 schema 的权威定义。

## 12. 扩展建议

如果你后续想扩展 `matrix_vis`，但又想保持当前结构清晰，建议遵循下面的边界：

- 新的用户工作流放到 `pipelines/`
- 新的文件契约放到 `io/`
- `cli/` 保持薄
- dataclass 契约集中在 `core/types.py`
- 几何和轻量数据变换放在 `core/`
- 优化问题装配和求解放在 `qp/`

这就是当前代码结构想表达的设计方向。

## 13. 最小 Python 接入示例

```python
from scripts.matrix_vis.pipelines.inspect import inspect_axis_config
from scripts.matrix_vis.pipelines.reconstruct import run_axis_reconstruction
from scripts.matrix_vis.pipelines.compose import run_motion_composition

inspect_axis_config("scripts/matrix_vis/configs/examples/axis_x_demo.yaml")

run_axis_reconstruction(
    config="scripts/matrix_vis/configs/examples/axis_x_demo.yaml",
)

run_axis_reconstruction(
    config="scripts/matrix_vis/configs/examples/axis_y_demo.yaml",
)

run_motion_composition(
    config="scripts/matrix_vis/configs/examples/compose_demo.yaml",
)
```

## 14. 建议优先阅读的源码文件

如果你还想再深入一层，建议按这个顺序看：

1. [README.md](/home/weizilin/generate_idea/scripts/matrix_vis/README.md)
2. [modeling.md](/home/weizilin/generate_idea/scripts/matrix_vis/modeling.md)
3. [pipelines/reconstruct.py](/home/weizilin/generate_idea/scripts/matrix_vis/pipelines/reconstruct.py:1)
4. [pipelines/compose.py](/home/weizilin/generate_idea/scripts/matrix_vis/pipelines/compose.py:1)
5. [io/config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/config.py:165)
6. [io/compose_config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/compose_config.py:45)
7. [core/types.py](/home/weizilin/generate_idea/scripts/matrix_vis/core/types.py:21)

## 15. 当前稳定性边界

这份文档描述的是“当前外部契约”，不是对长期最终 API 的承诺。

目前相对稳定的部分：

- YAML 配置结构
- dataclass 层的数据对象
- pipeline 函数名
- `solution.npz` 与 `composed_motion.npz` 的 key 布局

目前相对不稳定的部分：

- solver 内部实现
- `intersection` 之外的 compose policy
- 正则项权重在研究语境下的解释方式
