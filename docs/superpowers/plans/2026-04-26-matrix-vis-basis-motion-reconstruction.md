# Matrix Vis Basis Motion Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `scripts/matrix_vis` post-hoc analysis framework that reconstructs single-axis mesh trajectories from basis matrices, then composes `x` and `y` results into 2D motion previews.

**Architecture:** Keep the public surface small with three CLI entrypoints and drive everything from `yaml` configs. Internally split the work into `io`, `core`, `qp`, and `viz` modules so mesh loading, observation construction, convex QP solving, and visualization can evolve independently as real `disentangleNet` basis artifacts are introduced.

**Tech Stack:** Python, NumPy, pandas, PyYAML, cvxpy, OSQP, matplotlib or Pillow for static rendering, pytest for lightweight module tests

---

### Task 1: Scaffold the `matrix_vis` package layout

**Files:**
- Create: `scripts/matrix_vis/README.md`
- Create: `scripts/matrix_vis/__init__.py`
- Create: `scripts/matrix_vis/cli/__init__.py`
- Create: `scripts/matrix_vis/core/__init__.py`
- Create: `scripts/matrix_vis/io/__init__.py`
- Create: `scripts/matrix_vis/qp/__init__.py`
- Create: `scripts/matrix_vis/viz/__init__.py`
- Create: `scripts/matrix_vis/tests/__init__.py`
- Create: `scripts/matrix_vis/configs/examples/axis_x_demo.yaml`
- Create: `scripts/matrix_vis/configs/examples/axis_y_demo.yaml`
- Create: `scripts/matrix_vis/configs/examples/compose_demo.yaml`

- [ ] Step 1: Create the package directories and empty module markers so imports are stable from the first commit.
- [ ] Step 2: Write `README.md` with the three CLI entrypoints, expected workflow, and the current first-phase scope limits.
- [ ] Step 3: Add example config files that mirror the approved spec structure and use placeholder toy paths where real data is not available yet.
- [ ] Step 4: Commit the scaffold-only change set with a message such as `feat: scaffold matrix_vis package`.

### Task 2: Define stable config and domain types

**Files:**
- Create: `scripts/matrix_vis/core/types.py`
- Create: `scripts/matrix_vis/io/config.py`
- Create: `scripts/matrix_vis/tests/test_config.py`

- [ ] Step 1: Add dataclasses for `MeshTemplate`, `AxisProjection`, `BasisObservation`, `QPConfig`, `TrajectorySolution`, and `ComposedMotion`.
- [ ] Step 2: Implement config parsing from `yaml` into validated typed structures rather than passing nested dicts through the whole stack.
- [ ] Step 3: Validate required keys, supported mesh dimensions, supported axis labels, and anchor/subset consistency.
- [ ] Step 4: Write `scripts/matrix_vis/tests/test_config.py` to cover a valid minimal config and at least two invalid config cases.
- [ ] Step 5: Run `pytest scripts/matrix_vis/tests/test_config.py -v` and fix failures before moving on.
- [ ] Step 6: Commit the typed-config slice with a message such as `feat: add matrix_vis typed config loading`.

### Task 3: Add mesh loading and axis projection utilities

**Files:**
- Create: `scripts/matrix_vis/io/load_mesh.py`
- Create: `scripts/matrix_vis/core/mesh.py`
- Create: `scripts/matrix_vis/core/projection.py`
- Create: `scripts/matrix_vis/tests/test_projection.py`

- [ ] Step 1: Support loading a mesh from `.npy` first, with explicit dimension checks for 2D and 3D point arrays.
- [ ] Step 2: Implement projection from full mesh coordinates to a single axis using `source_axis_index`.
- [ ] Step 3: Implement subset extraction by global point IDs and preserve the mapping between global IDs and local subset order.
- [ ] Step 4: Add a projection test that verifies 3D-to-1D projection, subset extraction, and anchor lookup on toy coordinates.
- [ ] Step 5: Run `pytest scripts/matrix_vis/tests/test_projection.py -v`.
- [ ] Step 6: Commit the mesh/projection slice with a message such as `feat: add matrix_vis mesh projection utilities`.

### Task 4: Add basis loading and observation-table construction

**Files:**
- Create: `scripts/matrix_vis/io/load_basis.py`
- Create: `scripts/matrix_vis/core/basis.py`
- Create: `scripts/matrix_vis/core/observations.py`
- Create: `scripts/matrix_vis/tests/test_observations.py`

- [ ] Step 1: Load either a single square basis matrix or an indexed stack of square basis matrices from `.npy`.
- [ ] Step 2: Validate that the selected basis shape matches the subset-point count exactly.
- [ ] Step 3: Convert the basis matrix into a pairwise observation table with `(i, j, point_id_i, point_id_j, value)` rows, using one canonical triangle only.
- [ ] Step 4: Preserve room for optional pair filtering or weighting in the observation builder API without implementing every policy yet.
- [ ] Step 5: Write a toy test that verifies correct triangle extraction and point-ID alignment.
- [ ] Step 6: Run `pytest scripts/matrix_vis/tests/test_observations.py -v`.
- [ ] Step 7: Commit the basis/observation slice with a message such as `feat: add matrix_vis basis observation builder`.

### Task 5: Implement QP variable indexing and builder primitives

**Files:**
- Create: `scripts/matrix_vis/qp/variables.py`
- Create: `scripts/matrix_vis/qp/objective.py`
- Create: `scripts/matrix_vis/qp/constraints.py`
- Create: `scripts/matrix_vis/qp/builder.py`
- Create: `scripts/matrix_vis/tests/test_qp_builder.py`

- [ ] Step 1: Implement a deterministic variable-indexing helper mapping `(point_idx, time_idx)` to the flattened QP variable position.
- [ ] Step 2: Encode the data-fitting term from `modeling.md` using projected initial coordinates and observation values.
- [ ] Step 3: Encode the acceleration and velocity regularizers as separate weighted terms, even if both currently use second differences.
- [ ] Step 4: Encode initial-position equality constraints, anchor-point fixed constraints, ordering constraints, and optional displacement bounds.
- [ ] Step 5: Expose one builder entrypoint that returns a structured problem description or a ready-to-solve `cvxpy` problem plus metadata.
- [ ] Step 6: Add a unit test that verifies variable counts, constraint counts, and builder behavior on a tiny three-point toy case.
- [ ] Step 7: Run `pytest scripts/matrix_vis/tests/test_qp_builder.py -v`.
- [ ] Step 8: Commit the QP-builder slice with a message such as `feat: add matrix_vis qp builder`.

### Task 6: Integrate `cvxpy` + `osqp` solving and result serialization

**Files:**
- Create: `scripts/matrix_vis/qp/solve.py`
- Create: `scripts/matrix_vis/io/save_results.py`
- Modify: `scripts/matrix_vis/core/types.py`

- [ ] Step 1: Wrap the QP solve step behind one function that records solver status, objective value, iteration count, and residual diagnostics.
- [ ] Step 2: Define a standard `TrajectorySolution` serialization layout for `.npz` and `.json` outputs.
- [ ] Step 3: Save `resolved_config.yaml`, `summary.json`, `qp_diagnostics.json`, `solution.npz`, `observations.csv`, and `projected_mesh.csv` from one shared output helper.
- [ ] Step 4: Fail gracefully when `cvxpy` or `osqp` is unavailable by writing a readable error message rather than crashing deep in the stack.
- [ ] Step 5: Add a tiny end-to-end toy solve smoke test if feasible; otherwise add a builder-only smoke assertion and document the missing dependency path.
- [ ] Step 6: Commit the solver/output slice with a message such as `feat: add matrix_vis qp solve and export`.

### Task 7: Add CLI entrypoints for inspect and single-axis reconstruction

**Files:**
- Create: `scripts/matrix_vis/cli/inspect_config.py`
- Create: `scripts/matrix_vis/cli/reconstruct_axis.py`
- Modify: `scripts/matrix_vis/README.md`

- [ ] Step 1: Implement `inspect_config.py` to parse a config, load mesh and basis metadata, and print or save a dry-run validation report.
- [ ] Step 2: Implement `reconstruct_axis.py` to orchestrate config loading, projection, observation construction, QP solving, and result export.
- [ ] Step 3: Keep CLI arguments minimal: `--config`, `--axis`, and `--output_dir` overrides only.
- [ ] Step 4: Document example invocations in `README.md`.
- [ ] Step 5: Run `python scripts/matrix_vis/cli/inspect_config.py --config scripts/matrix_vis/configs/examples/axis_x_demo.yaml`.
- [ ] Step 6: Run `python scripts/matrix_vis/cli/reconstruct_axis.py --config scripts/matrix_vis/configs/examples/axis_x_demo.yaml` on toy data or a temporary generated fixture.
- [ ] Step 7: Commit the CLI slice with a message such as `feat: add matrix_vis axis reconstruction cli`.

### Task 8: Add single-axis diagnostic plots

**Files:**
- Create: `scripts/matrix_vis/viz/axis_plots.py`
- Modify: `scripts/matrix_vis/io/save_results.py`
- Modify: `scripts/matrix_vis/cli/reconstruct_axis.py`

- [ ] Step 1: Render a trajectory line plot showing each subset point across time.
- [ ] Step 2: Render a final-displacement plot that makes the recovered basis effect easy to scan.
- [ ] Step 3: Render a basis-fit error plot or summary figure using predicted versus observed pairwise average distance deltas.
- [ ] Step 4: Save plots conditionally from the `export` config block.
- [ ] Step 5: Run one toy reconstruction and verify that the three image files are emitted under the configured output directory.
- [ ] Step 6: Commit the diagnostic-plot slice with a message such as `feat: add matrix_vis axis diagnostics`.

### Task 9: Add 2D composition and preview rendering

**Files:**
- Create: `scripts/matrix_vis/cli/compose_motion.py`
- Create: `scripts/matrix_vis/viz/mesh_animation.py`
- Create: `scripts/matrix_vis/viz/exporters.py`
- Create: `scripts/matrix_vis/tests/test_compose_motion.py`

- [ ] Step 1: Load two previously saved single-axis solutions and verify compatible time grids and point-ID alignment.
- [ ] Step 2: Compose `x` and `y` trajectories into 2D coordinates for the shared subset, keeping other mesh points static in phase one.
- [ ] Step 3: Render at least one static snapshot and one simple animation artifact such as a gif or frame directory.
- [ ] Step 4: Save `composed_motion.npz` and `composed_summary.json` using the same output conventions as the axis pipeline.
- [ ] Step 5: Add a toy test covering point-ID alignment and 2D coordinate composition.
- [ ] Step 6: Run `pytest scripts/matrix_vis/tests/test_compose_motion.py -v`.
- [ ] Step 7: Run `python scripts/matrix_vis/cli/compose_motion.py --config scripts/matrix_vis/configs/examples/compose_demo.yaml`.
- [ ] Step 8: Commit the composition slice with a message such as `feat: add matrix_vis motion composition`.

### Task 10: Prepare the path for `disentangleNet` integration

**Files:**
- Modify: `scripts/matrix_vis/README.md`
- Modify: `docs/superpowers/specs/2026-04-26-matrix-vis-basis-motion-reconstruction-design.md`

- [ ] Step 1: Document the first supported external artifact contract for importing basis matrices from `disentangleNet` analysis outputs.
- [ ] Step 2: List the still-open questions: basis source layout, mesh-point conventions, and whether side/shared basis need different wrappers.
- [ ] Step 3: Add a short operator checklist for real-data onboarding so future work starts from a known sequence.
- [ ] Step 4: Commit the integration-notes slice with a message such as `docs: document matrix_vis integration handoff`.
