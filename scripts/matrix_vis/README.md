# matrix_vis

`scripts/matrix_vis` is the post-hoc analysis package for turning a single-axis
window-difference matrix into an interpretable local face-mesh motion
reconstruction.

The current package should be read as an inverse-problem sandbox, not as a
finished "true basis to true trajectory" solver. Its job is to make the motion
assumptions explicit, run a constrained reconstruction, and emit diagnostics
that show how plausible that reconstruction is.

## Problem Background

The intended background is:

- The data is split into fixed-length windows.
- For each window and each axis (`x` or `y`), a `window matrix` records the
  average pairwise distance behavior of the points inside that window.
- A `window diff` is the matrix difference between two adjacent window
  matrices.
- In the current inverse setting, we assume the previous window is the known
  reference state.
- We know the point positions at the start of that reference window.
- We then try to reconstruct the motion trajectory in the next window from the
  `window diff`, under additional smoothness assumptions.

Stated more concretely:

1. A window length fixes the number of frames to reconstruct.
2. The input is a single-axis window-difference matrix.
3. The previous window is treated as static / known.
4. The next window is the unknown trajectory we want to recover.

This framing is the working assumption behind
[`modeling.md`](/home/weizilin/generate_idea/scripts/matrix_vis/modeling.md).

## Why The Solver Reconstructs A Trajectory

The observation does not uniquely determine a full frame-by-frame trajectory.
That is expected.

The current design still reconstructs a time sequence because:

- the window length is known in frames
- the goal is not only to estimate a window-level displacement summary
- the downstream use case is to render a plausible motion clip inside that
  window

So the optimization problem should be understood as:

> find one smooth, constraint-compatible trajectory that explains the observed
> window difference, rather than recover the only physically correct trajectory

This is why the package uses a QP-style reconstruction instead of a direct
closed-form inversion.

## Current Working Assumptions

These are the assumptions we are currently keeping:

- single-axis reconstruction first, then optional `x` / `y` composition
- previous-window positions are known and used as initial positions
- the input observation is treated as a window-difference target
- the solver may use regularization to choose one plausible trajectory among
  many underdetermined solutions
- structural consistency diagnostics should be reported, because not every
  symmetric matrix is guaranteed to come from a realizable trajectory

## Known Issues And Clarifications

The following points are already acknowledged and should be treated as active
design notes rather than hidden caveats:

- The observation alone cannot determine the full trajectory. This is why the
  current method is explicitly optimization-based.
- Reconstructing a full time sequence is still intentional because the window
  frame count is given and the desired output is a within-window motion
  trajectory, not just a displacement summary.
- The early draft relied too strongly on a "point order stays unchanged"
  assumption. This is a real issue and still needs improvement.
- The old idea of enforcing a simple order constraint from point IDs was too
  strong and has already been abandoned. `modeling.md` should be read as a
  draft, not as the final accepted constraint set.
- The observation matrix may fail realizability / consistency conditions. A
  future version should add an explicit structural-residual style diagnostic so
  we can distinguish "good fit to the assumed inverse model" from "badly
  realizable input".
- The meaning of `lambda_acc` and `lambda_vel` is not yet cleanly aligned
  between the write-up and the implementation. This is a known problem.
- The toy pipeline only proves that the algorithm stack can run end to end. It
  is not meant to prove that the real inverse problem assumptions are fully
  valid.
- Basis amplitude calibration is out of scope for the current stage. That topic
  should not drive the present `matrix_vis` design decisions yet.

## Phase-1 Scope

Phase 1 remains intentionally narrow:

- reconstruct one axis at a time (`x` or `y`) from a standard mesh, a point
  subset, one anchor point, and one basis / observation matrix
- drive runs from `yaml` configs, with only a few CLI overrides
- save intermediate artifacts and diagnostics for inspection
- compose previously solved `x` and `y` results into a 2D preview in a later
  step

## CLI Entrypoints

- `python scripts/matrix_vis/cli/inspect_config.py --config <path>`
  - dry-run input validation only
- `python scripts/matrix_vis/cli/reconstruct_axis.py --config <path>`
  - single-axis reconstruction workflow
- `python scripts/matrix_vis/cli/compose_motion.py --config <path>`
  - combine solved `x` and `y` trajectories into a 2D preview

## Example Workflow

1. Prepare a standard face-mesh template in 2D or 3D.
2. Select the subset of point IDs covered by the matrix.
3. Choose one fixed anchor point inside that subset.
4. Run one axis config for `x`.
5. Run one axis config for `y`.
6. Compose the two solved trajectories into a 2D motion preview.

## Toy Data

The example configs in `configs/examples/` target a toy setup:

- mesh shape: a leaf-like closed contour that opens toward a near-rectangular
  mouth-like target
- motion pattern: an opening-mouth trajectory
- purpose: verify that config loading, QP solving, export, and composition can
  run end to end

The toy assets are generated by:

```bash
python scripts/matrix_vis/scripts/generate_toy_double_crescent_data.py
```

This writes mesh, trajectory, and single-axis basis files under:

- `data/toy/matrix_vis/leaf_to_rectangle_mouth_opening/`

The toy generator is intentionally idealized. It should be treated as a
pipeline sanity check, not as evidence that the real inverse assumptions are
already validated.

## Current Scope Limits

- No direct `disentangleNet` checkpoint import yet.
- No propagation from a local subset to the full face.
- No multi-basis joint solve.
- No interactive UI.
- No promise that the compose-config schema is final before implementation
  lands.

## Layout

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

For the approved design and phase breakdown, see:

- `docs/superpowers/specs/2026-04-26-matrix-vis-basis-motion-reconstruction-design.md`
- `docs/superpowers/plans/2026-04-26-matrix-vis-basis-motion-reconstruction.md`
