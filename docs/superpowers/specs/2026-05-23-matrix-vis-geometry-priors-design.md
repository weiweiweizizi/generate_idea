# Matrix Vis Geometry Priors Design

Date: 2026-05-23

## Goal

Strengthen `scripts/matrix_vis` axis reconstruction with geometry-aware priors that operate on the existing coordinate-based solver, without introducing distance-only latent variables or a separate structural stage.

## Accepted Design

### 1. Topology source

- Copy MediaPipe face mesh connectivity into the repo as `scripts/matrix_vis/face_mesh_connections.py`.
- Use `FACEMESH_TESSELATION` as the canonical topology source.
- Filter the global topology down to the active `subset_point_ids`.

### 2. Gap-2 Laplacian regularization

- Keep the optimization variable as the dynamic axis trajectory `x_i(t)` or `y_i(t)`.
- Define the dynamic field on graph nodes as:
  - `u_i(t) = x_i(t+2) - x_i(t)` for `x`
  - `u_i(t) = y_i(t+2) - y_i(t)` for `y`
- Penalize spatial inconsistency of this field with the subset graph Laplacian:

  `sum_t ||L u_t||^2`

- Implement this as a quadratic regularizer added directly into `reg_p`, so the existing matrix-free linearized solver path can reuse it efficiently.

### 3. Oriented area sign barrier

- Build local triangles from the filtered face-mesh topology.
- At each time step, combine:
  - the solved dynamic axis, and
  - the static orthogonal axis from the canonical face template
  into a 2D embedding.
- Compute each triangle's oriented double area and compare it with the reference sign from the initial template state.
- Penalize sign flips and near-collapses with a smooth barrier:

  `softplus(margin - signed_area / reference_area_scale)^2`

- This is a soft barrier, not an area-preservation objective. Its purpose is to prevent local fold-overs, not to freeze deformation magnitude.

### 4. Solver routing

- If `lambda_area_sign == 0`, keep the fast quadratic path:
  - unconstrained: CG
  - bounded: LBFGS with box parameterization
- If `lambda_area_sign > 0`, route the inner solve through LBFGS so the non-linear barrier is actually optimized.

## Config additions

Add optional solver fields:

- `lambda_laplacian`
- `lambda_area_sign`
- `area_barrier_margin`
- `geometry_topology_source`

Defaults keep existing configs backward compatible.

## Implementation boundary

- `qp/geometry.py`: topology loading, subset filtering, triangle recovery, graph Laplacian construction
- `qp/builder.py`: geometry preprocessing, Laplacian quadratic assembly, area reference statistics
- `qp/solve.py`: area-sign barrier and solver-path selection
- `pipelines/reconstruct.py` and `pipelines/patient_sequence.py`: pass static orthogonal coordinates into `build_axis_qp`

## Verification

- Update unit tests for config parsing, QP bundle construction, and solver execution with geometry priors enabled.
