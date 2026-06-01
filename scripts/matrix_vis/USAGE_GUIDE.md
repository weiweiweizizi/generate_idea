# matrix_vis External Usage Guide

`matrix_vis` is a post-hoc reconstruction tool for turning a single-axis window-difference distance matrix into an interpretable local mesh motion trajectory.

This guide is written for external users who want to run the tool, understand its file contracts, and integrate it into a larger research workflow without reading all source files first.

## 1. What `matrix_vis` Does

Current `main` should be understood as:

- a single-axis (`x` or `y`) reconstruction workflow
- driven by YAML configs
- using a mesh template, a point subset, and a square observation matrix
- solving for a smooth within-window trajectory
- optionally composing solved `x` and `y` trajectories into a 2D animation preview

It is not a general-purpose package yet. It is a research sandbox with stable enough internal contracts to be used from scripts and reproducible configs.

## 2. Project Structure

```text
scripts/matrix_vis/
  README.md
  modeling.md
  USAGE_GUIDE.md
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

The directories are intended to be read like this:

- `cli/`
  - thin entrypoints exposed to users
  - should only parse command-line arguments and forward into pipelines
- `pipelines/`
  - end-to-end orchestration layer
  - this is the best place to read the actual workflow
- `io/`
  - config parsing
  - mesh loading
  - basis / matrix loading
  - result saving
- `core/`
  - stable data objects and lightweight transforms
  - projection from mesh to one axis
  - conversion from square matrix to pairwise observation table
- `qp/`
  - optimization problem assembly and solving
- `viz/`
  - plots, snapshots, frame export, GIF export
- `configs/examples/`
  - minimal toy examples
- `configs/real/`
  - real experiment configs used in this repository
- `tests/`
  - regression checks for config parsing, observation loading, projection, and QP assembly

## 3. Main User Entry Points

### 3.1 Inspect an axis config

```bash
python scripts/matrix_vis/cli/inspect_config.py inspect --config scripts/matrix_vis/configs/examples/axis_x_demo.yaml
```

What it does:

- loads the YAML config
- resolves mesh and basis inputs
- computes the projected subset
- checks the shape contract between subset and basis matrix
- prints a summary JSON to stdout

Use this before running a solve on a new config.

### 3.2 Reconstruct one axis

```bash
python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct --config scripts/matrix_vis/configs/examples/axis_x_demo.yaml
python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct --config scripts/matrix_vis/configs/examples/axis_y_demo.yaml
```

Optional override:

```bash
python scripts/matrix_vis/cli/reconstruct_axis.py reconstruct \
  --config path/to/config.yaml \
  --output_dir outputs/matrix_vis/custom_run
```

What it does:

- loads config
- loads mesh
- projects the mesh to `x` or `y`
- loads a basis matrix or a `next - prev` matrix difference
- converts the square matrix to pairwise observations
- assembles the QP bundle
- solves the trajectory
- exports plots, summaries, and `solution.npz`

### 3.3 Compose solved `x` and `y` results

```bash
python scripts/matrix_vis/cli/compose_motion.py compose --config scripts/matrix_vis/configs/examples/compose_demo.yaml
```

What it does:

- loads one mesh
- loads `solution.npz` from a solved `x` run
- loads `solution.npz` from a solved `y` run
- intersects their point sets
- writes the reconstructed `x/y` coordinates back into the mesh over time
- exports a snapshot, optional GIF, and `composed_motion.npz`

## 4. End-to-End Dataflow

This is the current operational dataflow.

### 4.1 Axis reconstruction dataflow

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

### 4.2 Motion composition dataflow

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

## 5. Core Concepts and Contracts

External users should keep these concepts separate.

### 5.1 Mesh

A mesh is the geometric template from which the solver takes initial positions.

Current supported input formats:

- `numpy`
  - usually `.npy`
  - expected to contain `[N, D]`
- `mediapipe_canonical_obj`
  - canonical face mesh OBJ
  - loader reads the first 468 vertices
  - can synthesize iris points to get 478 points

Current supported dimensions:

- `2d`
- `3d`

### 5.2 Projection

Reconstruction is always performed on one axis at a time.

The projection step turns the mesh into:

- `full_axis_positions`
  - all points on the selected source axis
- `subset_point_ids`
  - the point IDs actually involved in the inverse problem
- `subset_positions`
  - initial 1D positions for the selected subset
- `anchor_point_ids`
  - points held fixed during the whole trajectory

### 5.3 Basis observation

The observation is currently a square matrix with semantic:

\[
\Delta D_{ij}
=
\frac{1}{T}\sum_t |x_j(t)-x_i(t)| - |x_j(0)-x_i(0)|
\]

Current supported semantics:

- `mean_distance_delta`

The tool supports two ways to provide the matrix:

- `basis.source`
  - direct `.npy` matrix or matrix stack
- `basis.prev_source` + `basis.next_source`
  - the loader will compute `next - prev`

### 5.4 Observation table

The square matrix is converted to an upper-triangle pairwise table:

- `i`
  - local subset index
- `j`
  - local subset index
- `point_id_i`
  - global point ID
- `point_id_j`
  - global point ID
- `value`
  - observed matrix entry

This table is the direct input to QP construction.

## 6. Stable Python Interfaces

These are the most useful interfaces if you want to call `matrix_vis` from your own Python code.

### 6.1 Config dataclasses

Defined in [core/types.py](/home/weizilin/generate_idea/scripts/matrix_vis/core/types.py:21):

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

These represent the most stable external contract layer.

### 6.2 Loaded data dataclasses

Also defined in [core/types.py](/home/weizilin/generate_idea/scripts/matrix_vis/core/types.py:123):

- `MeshTemplate`
- `AxisProjection`
- `BasisObservation`
- `TrajectorySolution`
- `ComposedMotion`

### 6.3 Pipeline functions

The most practical callable entry points are:

- `scripts.matrix_vis.pipelines.inspect.inspect_axis_config`
- `scripts.matrix_vis.pipelines.reconstruct.run_axis_reconstruction`
- `scripts.matrix_vis.pipelines.compose.run_motion_composition`

These are preferable to calling the CLI modules from Python.

## 7. Axis Config Schema

The axis config is parsed by [io/config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/config.py:165).

Minimal shape:

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
  qp_backend: torch

export:
  save_projected_mesh: true
  save_qp_diagnostics: true
  save_axis_plot: true
  save_npz: true
  save_json_summary: true
```

### 7.1 `experiment`

- `name: str`
  - display name used in summaries
- `output_dir: str`
  - output folder for this run

### 7.2 `mesh`

- `source: str`
  - path to mesh input
- `format: str`
  - one of `numpy`, `mediapipe_canonical_obj`
- `dimension: str`
  - one of `2d`, `3d`
- `point_ids: auto | [int, ...]`
  - `auto` means assign `0..N-1`
  - explicit list must match point count
- `normalization_scope: str | null`
  - optional
  - one of `mouth_only`, `eye_only`, `face_regions`

### 7.3 `projection`

- `axis: str`
  - one of `x`, `y`
- `source_axis_index: int`
  - usually `0` for `x`, `1` for `y`
- `subset_point_ids: [int, ...]`
  - explicit subset to reconstruct
- `anchor_point_ids: [int, ...]`
  - one or more fixed points

Alternative subset specification:

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

Current supported `subset_layout.name`:

- `face_regions_grouped`
- `mouth`

Validation rules:

- subset IDs must be unique
- anchor IDs must be unique
- every anchor ID must be included in the subset
- source axis index must match mesh dimension

### 7.4 `basis`

Direct matrix mode:

```yaml
basis:
  source: path/to/basis_x.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta
```

Difference mode:

```yaml
basis:
  prev_source: path/to/window_prev.npy
  next_source: path/to/window_next.npy
  basis_index: 0
  matrix_shape: square
  value_semantics: mean_distance_delta
```

Optional full-matrix crop mode:

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

Use `matrix_layout` when:

- your loaded matrix covers a larger canonical ordering
- but the current run only reconstructs a subset
- and you need `matrix_vis` to crop the square matrix consistently

### 7.5 `solver`

- `num_time_steps: int`
  - must be `>= 2`
- `lambda_data: float`
  - data term weight
- `lambda_acc: float`
  - second-difference temporal smoothness weight
- `lambda_vel: float`
  - velocity fluctuation regularization weight
- `enforce_order: bool`
  - currently parsed but not the main practical control path
- `max_displacement: float | null`
  - optional hard bound around initial positions
- `qp_backend: str`
  - `torch`
- `max_observations: int | null`
  - optional truncation of largest-magnitude pairwise observations

### 7.6 `export`

- `save_projected_mesh: bool`
- `save_qp_diagnostics: bool`
- `save_axis_plot: bool`
- `save_npz: bool`
- `save_json_summary: bool`

## 8. Compose Config Schema

The compose config is parsed by [io/compose_config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/compose_config.py:45).

Current supported schema:

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

Important current limitation:

- only `subset_policy: intersection` is supported

That means:

- the composed motion only uses points that exist in both `x` and `y` solutions
- other mesh points remain static because the pipeline initializes coordinates from the original mesh and only overwrites the intersected subset

## 9. Output Contracts

### 9.1 Axis reconstruction outputs

Typical files:

- `resolved_config.yaml`
- `projected_mesh.csv`
- `observations.csv`
- `solution.npz`
- `summary.json`
- `qp_diagnostics.json`
- `axis_trajectory.png`
- `axis_ground_truth_comparison.png` when toy ground truth exists

#### `projected_mesh.csv`

Columns:

- `point_id`
- `axis_position`

#### `observations.csv`

Columns:

- `i`
- `j`
- `point_id_i`
- `point_id_j`
- `value`

#### `solution.npz`

Keys:

- `point_ids`
- `time_grid`
- `initial_positions`
- `trajectory`
- `anchor_point_ids`
- `anchor_point_id`
- `basis_matrix`

Shapes:

- `point_ids`: `[N]`
- `time_grid`: `[T]`
- `initial_positions`: `[N]`
- `trajectory`: `[N, T]`
- `anchor_point_ids`: `[K]`
- `basis_matrix`: `[N, N]`

#### `summary.json`

Typical fields:

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

### 9.2 Composed motion outputs

Typical files:

- `motion_snapshot.png`
- `frames/`
- `motion_preview.gif`
- `composed_motion.npz`
- `composed_summary.json`

#### `composed_motion.npz`

Keys:

- `point_ids`
- `time_grid`
- `coordinates`
- `subset_point_ids`

Shapes:

- `point_ids`: `[N]`
- `time_grid`: `[T]`
- `coordinates`: `[T, N, D]`
- `subset_point_ids`: `[M]`

## 10. Recommended Usage Pattern

For a new dataset or experiment, use this order.

1. Prepare a mesh with stable point IDs.
2. Decide which axis to reconstruct first.
3. Decide the subset and anchors.
4. Verify that the basis matrix ordering matches the subset ordering.
5. Run `inspect`.
6. Run one axis solve.
7. Inspect `summary.json`, `qp_diagnostics.json`, and plots.
8. Repeat for the second axis if needed.
9. Run `compose`.
10. Inspect `motion_snapshot.png` and `motion_preview.gif`.

## 11. Common Failure Modes

### 11.1 Basis shape does not match subset point count

Typical cause:

- subset IDs and matrix ordering are inconsistent
- you loaded a full-region matrix but configured a local subset

Fix:

- align the basis matrix to the subset directly
- or provide `basis.matrix_layout` so the loader can crop by point ID

### 11.2 Anchor point not in subset

Typical cause:

- config typo
- subset changed but anchors were not updated

Fix:

- ensure every `anchor_point_id` is included in `subset_point_ids`

### 11.3 Wrong `source_axis_index`

Typical cause:

- `x/y` label and axis index do not match
- using `2` on a 2D mesh

Fix:

- use `0` for `x`
- use `1` for `y`
- only use `2` when the mesh dimension is `3d`

### 11.4 No overlap between `x` and `y` solutions during compose

Typical cause:

- the two runs used different subsets
- the saved `solution.npz` files came from incompatible configs

Fix:

- ensure the intended compose set appears in both axis runs

### 11.5 Example config contains fields not supported by current code

If you are copying old configs from this repo history, re-check them against:

- [io/config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/config.py:165)
- [io/compose_config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/compose_config.py:45)

Those two files are the authoritative schema definitions.

## 12. Extension Points

If you want to extend `matrix_vis` while preserving current structure:

- add new user workflows under `pipelines/`
- add new file contracts under `io/`
- keep `cli/` thin
- keep dataclass contracts in `core/types.py`
- keep geometry transforms in `core/`
- keep optimization assembly in `qp/`

This separation is the current intended architecture.

## 13. Minimal Python Integration Example

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

## 14. Source Files Worth Reading First

If you need to go one level deeper, read in this order:

1. [README.md](/home/weizilin/generate_idea/scripts/matrix_vis/README.md)
2. [modeling.md](/home/weizilin/generate_idea/scripts/matrix_vis/modeling.md)
3. [pipelines/reconstruct.py](/home/weizilin/generate_idea/scripts/matrix_vis/pipelines/reconstruct.py:1)
4. [pipelines/compose.py](/home/weizilin/generate_idea/scripts/matrix_vis/pipelines/compose.py:1)
5. [io/config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/config.py:165)
6. [io/compose_config.py](/home/weizilin/generate_idea/scripts/matrix_vis/io/compose_config.py:45)
7. [core/types.py](/home/weizilin/generate_idea/scripts/matrix_vis/core/types.py:21)

## 15. Current Scope Boundary

This guide documents the current external contract, not a promise of final long-term API stability.

The most stable parts today are:

- YAML config structure
- dataclass-level data objects
- pipeline function names
- `solution.npz` and `composed_motion.npz` key layout

The less stable parts today are:

- solver internals
- extra compose policies beyond `intersection`
- research-specific interpretation of regularization weights
