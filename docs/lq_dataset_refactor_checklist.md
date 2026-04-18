# LQ Dataset Refactor Checklist

Last updated: 2026-04-18

## Context

Current `scripts/lq/datasets.py` is a minimal dataset implementation for the LQ disentanglement prototype. It is already usable for smoke tests, but it does not yet fully encode the actual research data semantics used in `data/win20-step20/TT` and `data/win20-step20/IMR`.

This checklist records the agreed refactor priorities before making larger changes to the training pipeline.

## Current Gaps

1. Region definitions are duplicated across files and should be unified.
2. `mode=x|y` currently only switches the loaded file suffix, but does not yet enforce direction-specific sample validity logic.
3. `deleted_x` / `deleted_y` in `metadata.csv` are not yet used when constructing valid diff samples.
4. Dataset preprocessing and action-basis initialization do not yet share all structural assumptions in one place.
5. Sample metadata is still minimal for downstream analysis.
6. Sampling remains window-flat and is not yet patient-balanced or group-balanced.

## Agreed Priorities

### Phase 1: Do Soon

1. Unify region definitions
   - Use one shared definition for `full` and `mouth`.
   - `mouth` should follow the agreed crop range `188:307`.
   - Reuse the same region definition in dataset loading, basis initialization, and future visualization scripts.

2. Clarify direction-specific dataset semantics
   - `mode=x` and `mode=y` should eventually become true direction-specific sample builders, not just different filenames.
   - This matters especially once `deleted_x` / `deleted_y` filtering is added.

3. Add `deleted_x` / `deleted_y` filtering
   - Source folders: `data/win20-step20/TT` and `data/win20-step20/IMR`
   - Their `metadata.csv` contains `deleted_x` and `deleted_y`.
   - Intended rule for later implementation:
     - `mode=x`: skip samples whose current diff window is marked `deleted_x`
     - `mode=y`: skip samples whose current diff window is marked `deleted_y`
   - This item is intentionally deferred for now, but it is the next important semantic fix.

4. Align dataset structure priors with basis initialization
   - Action basis initialization already applies:
     - symmetry enforcement
     - zero diagonal
     - region crop
   - Dataset preprocessing should later support optional alignment with these assumptions.

5. Improve per-sample metadata
   - Keep or add stable fields such as:
     - `subject`
     - `dataset_name`
     - `window_idx`
     - `mode`
     - `sample_id`
   - This is needed for later analysis of code activation and reconstruction behavior.

6. Clarify label layering
   - Primary supervision: `side_label`
   - Auxiliary supervision: `severity_label`, `dataset_label`
   - Raw metadata: `label_5class`, `score`

### Phase 2: Do Later

1. Patient-balanced sampling
   - Avoid patients with many windows dominating optimization.

2. Dataset-balanced or side-balanced sampling
   - Useful if IMR/TT or side distributions bias training.

3. Optional raw-pair outputs
   - Return `current_matrix`, `prev_matrix`, and `delta_matrix` when needed for analysis.

4. Future multi-branch extensibility
   - Keep room for later `xy` or dual-branch input modes.

5. Richer group metadata
   - Explicit grouped fields for source / side / severity if later losses or analysis require them.

6. Caching / acceleration
   - Consider only after dataset semantics are stable.

## Recommended Execution Order

1. Extract shared region constants / utilities.
2. Improve dataset metadata interface.
3. Add direction-aware `deleted_x` / `deleted_y` filtering.
4. Revisit matrix-structure alignment and sampling strategy.

## Notes

- For now, `deleted_x` / `deleted_y` handling is intentionally postponed.
- The immediate focus remains keeping the current pipeline runnable while gradually aligning it with the actual research semantics.
