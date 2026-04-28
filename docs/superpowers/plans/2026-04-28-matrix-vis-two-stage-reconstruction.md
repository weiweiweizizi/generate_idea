# Matrix Vis Two-Stage Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `matrix_vis` reconstruction core with the confirmed two-stage algorithm: first project a window diff into a realizable structural target, then generate a monotone window trajectory from a small set of shared time bases.

**Architecture:** Keep the existing toy-driven `matrix_vis` framework, CLI, and config surface intact. Only change the algorithmic core in `core/`, `qp/`, and the algorithm-related export path so Stage A structural projection, Stage C trajectory generation, and diagnostics can evolve independently without dragging in `disentangleNet` integration work.

**Tech Stack:** Python, NumPy, pandas, cvxpy, OSQP, pytest

---

## File Map

- Create: `scripts/matrix_vis/core/structure.py`
  - Stage A helpers for pairwise structure recovery, projection residuals, and structural diagnostics.
- Modify: `scripts/matrix_vis/qp/objective.py`
  - Replace the old single-stage objective with Stage A / Stage C terms.
- Modify: `scripts/matrix_vis/qp/constraints.py`
  - Remove the old ID-order assumption and add the new sign-consistency / anchor constraints.
- Modify: `scripts/matrix_vis/qp/builder.py`
  - Build the two-stage reconstruction bundle and keep the existing toy entrypoints stable.
- Modify: `scripts/matrix_vis/qp/solve.py`
  - Return Stage A / Stage C diagnostics together with the solved trajectories.
- Modify: `scripts/matrix_vis/io/save_results.py`
  - Persist structural residuals and the new diagnostic summaries.
- Modify: `scripts/matrix_vis/cli/reconstruct_axis.py`
  - Orchestrate the new two-stage core without changing the CLI surface.
- Modify: `scripts/matrix_vis/modeling.md`
  - Keep the algorithm write-up aligned with the implemented two-stage flow.
- Create: `scripts/matrix_vis/tests/test_structure.py`
  - Unit tests for structural projection and residuals.
- Create: `scripts/matrix_vis/tests/test_two_stage_reconstruction.py`
  - End-to-end toy regression tests for the new algorithmic path.
- Modify: `scripts/matrix_vis/tests/test_qp_builder.py`
  - Update the builder expectations to match the new formulation.

---

### Task 1: Add Stage A structural projection primitives

**Files:**
- Create: `scripts/matrix_vis/core/structure.py`
- Create: `scripts/matrix_vis/tests/test_structure.py`

- [ ] **Step 1: Write the failing structural-projection test**

```python
import numpy as np

from scripts.matrix_vis.core.structure import project_window_structure


def test_project_window_structure_returns_projection_and_residual():
    x0 = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    d_raw = np.asarray(
        [
            [0.0, 0.9, 1.8],
            [0.9, 0.0, 0.8],
            [1.8, 0.8, 0.0],
        ],
        dtype=np.float32,
    )

    result = project_window_structure(x0, d_raw)

    assert result.z.shape == (3,)
    assert result.d_hat.shape == (3, 3)
    assert result.structural_residual.shape == (3, 3)
    assert result.structural_residual_rmse >= 0.0
```

- [ ] **Step 2: Run the test to confirm it fails first**

Run:

```bash
pytest scripts/matrix_vis/tests/test_structure.py -v
```

Expected: fail because `project_window_structure(...)` does not exist yet.

- [ ] **Step 3: Implement the minimal Stage A projection**

Add `project_window_structure(...)` and a small result dataclass that:

```python
result = project_window_structure(x0, d_raw)
```

should produce:

- `z`: the projected structural target
- `d_hat`: the realizable pairwise matrix
- `structural_residual`: `d_hat - d_raw`
- `structural_residual_rmse`

Use a simple, explicit optimization or projection formulation that stays compatible with the current toy assumptions.

- [ ] **Step 4: Run the test again**

Run:

```bash
pytest scripts/matrix_vis/tests/test_structure.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/matrix_vis/core/structure.py scripts/matrix_vis/tests/test_structure.py
git commit -m "matrix_vis: add stage a structural projection"
```

### Task 2: Replace the reconstruction core with Stage C shared time bases

**Files:**
- Modify: `scripts/matrix_vis/qp/objective.py`
- Modify: `scripts/matrix_vis/qp/constraints.py`
- Modify: `scripts/matrix_vis/qp/builder.py`
- Modify: `scripts/matrix_vis/qp/solve.py`
- Modify: `scripts/matrix_vis/tests/test_qp_builder.py`

- [ ] **Step 1: Write the failing Stage C builder test**

```python
import numpy as np
import pandas as pd

from scripts.matrix_vis.core.structure import project_window_structure
from scripts.matrix_vis.qp.builder import build_two_stage_axis_qp


def test_build_two_stage_axis_qp_uses_shared_monotone_basis():
    x0 = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    d_raw = np.asarray(
        [
            [0.0, 0.9, 1.8],
            [0.9, 0.0, 0.8],
            [1.8, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    stage_a = project_window_structure(x0, d_raw)
    observations = pd.DataFrame(
        [
            {"i": 0, "j": 1, "point_id_i": 10, "point_id_j": 11, "value": 0.9},
            {"i": 1, "j": 2, "point_id_i": 11, "point_id_j": 12, "value": 0.8},
        ]
    )

    bundle = build_two_stage_axis_qp(
        subset_point_ids=np.asarray([10, 11, 12], dtype=np.int64),
        initial_positions=x0,
        stage_a_result=stage_a,
        observations=observations,
        num_time_steps=25,
    )

    assert bundle.time_basis.shape[0] in {2, 3, 4}
    assert bundle.problem is not None
```

- [ ] **Step 2: Run the test to confirm it fails first**

Run:

```bash
pytest scripts/matrix_vis/tests/test_qp_builder.py -v
```

Expected: fail because the new two-stage builder and time-basis path are not implemented yet.

- [ ] **Step 3: Implement the two-stage reconstruction assembly**

Add the new builder path and refactor the objective / constraints so that:

```python
bundle = build_two_stage_axis_qp(...)
```

constructs:

- Stage A projected structure
- Stage C coefficients over `2~4` global shared monotone bases
- anchor constraints
- sign-consistency constraints
- local spatial smoothness
- no point-ID order constraint

Keep `x` and `y` solving independent, but use the same time-basis family on both axes.

- [ ] **Step 4: Run the test again**

Run:

```bash
pytest scripts/matrix_vis/tests/test_qp_builder.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/matrix_vis/qp/objective.py scripts/matrix_vis/qp/constraints.py scripts/matrix_vis/qp/builder.py scripts/matrix_vis/qp/solve.py scripts/matrix_vis/tests/test_qp_builder.py
git commit -m "matrix_vis: switch to two-stage reconstruction core"
```

### Task 3: Wire diagnostics and toy regression coverage

**Files:**
- Modify: `scripts/matrix_vis/io/save_results.py`
- Modify: `scripts/matrix_vis/cli/reconstruct_axis.py`
- Modify: `scripts/matrix_vis/modeling.md`
- Create: `scripts/matrix_vis/tests/test_two_stage_reconstruction.py`

- [ ] **Step 1: Write the failing end-to-end toy regression**

```python
from scripts.matrix_vis.cli.reconstruct_axis import reconstruct


def test_toy_reconstruction_emits_structural_diagnostics():
    summary = reconstruct("scripts/matrix_vis/configs/examples/axis_x_demo.yaml")

    assert "diagnostics" in summary
    assert "structural_residual_rmse" in summary["diagnostics"]
    assert summary["diagnostics"]["structural_residual_rmse"] >= 0.0
```

- [ ] **Step 2: Run the regression to confirm it fails first**

Run:

```bash
pytest scripts/matrix_vis/tests/test_two_stage_reconstruction.py -v
```

Expected: fail because the new diagnostics are not yet wired through the export path.

- [ ] **Step 3: Implement diagnostic export and CLI summary fields**

Persist and expose at least:

- `D_raw`
- `D_hat`
- `structural_residual_rmse`
- `trajectory_fit_rmse`
- `mean_alignment_rmse`
- `sign_conflict_count`

Keep the existing toy CLI workflow intact and do not change the CLI surface.

- [ ] **Step 4: Run the regression again**

Run:

```bash
pytest scripts/matrix_vis/tests/test_two_stage_reconstruction.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full matrix_vis test set**

Run:

```bash
pytest scripts/matrix_vis/tests -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/matrix_vis/io/save_results.py scripts/matrix_vis/cli/reconstruct_axis.py scripts/matrix_vis/modeling.md scripts/matrix_vis/tests/test_two_stage_reconstruction.py
git commit -m "matrix_vis: add structural diagnostics and toy regression"
```

