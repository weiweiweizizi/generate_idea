# DisentangleNet Matrix Vis Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge `scripts/disentangleNet` and `scripts/matrix_vis` so frozen `v31` `x/mouth` basis artifacts can drive post-hoc trajectory reconstruction, first per-basis with fixed `y` composition, then per-patient with coefficient-composed `x` observations and rolling-window visualization.

**Architecture:** Keep `matrix_vis` solver semantics unchanged: it still consumes one square observation matrix per single-axis reconstruction. Put the new logic into two thin bridge layers: `disentangleNet` exports basis stacks, patient coefficients, and composed per-window `x` observation matrices; `matrix_vis` adds config generation and a patient-sequence pipeline that can override initial `x` positions between consecutive windows while keeping `y` static or externally supplied.

**Tech Stack:** Python, NumPy, pandas, PyTorch, PyYAML, existing `scripts/disentangleNet` analysis helpers, existing `scripts/matrix_vis` CLI/pipeline stack, pytest for config/pipeline helpers, script-level smoke runs for real-data validation

---

### Task 1: Freeze the bridge artifact contract

**Files:**
- Modify: `scripts/disentangleNet/README.md`
- Modify: `scripts/matrix_vis/README.md`
- Modify: `scripts/matrix_vis/USAGE_GUIDE_CN.md`
- Create: `docs/disentanglenet_matrix_vis_contract.md`

- [ ] Step 1: Document the exact bridge scope in one place: Step 1 uses one exported `x` basis matrix at a time; Step 2 uses one patient’s window-level coefficient-composed `x` observation matrices.
- [ ] Step 2: Define the exported artifact layout under `outputs/disentangleNet/<run>/matrix_vis_exports/`, including filenames, shapes, dtypes, and how point ordering maps to `matrix_vis` subset layouts.
- [ ] Step 3: Explicitly record that `disentangleNet` basis artifacts are `mouth`-cropped `119 x 119` signed `mean_distance_delta` matrices and that `matrix_vis` will treat them as single-axis observations rather than latent basis vectors.
- [ ] Step 4: Record the patient-sequence rule for `TT/844697`: first window is static reference, each subsequent window starts from the previous reconstructed final frame, and `y` stays static in the first implementation.
- [ ] Step 5: Add a short operator checklist covering which checkpoint, which patient, which fixed `y` config, and where outputs land.
- [ ] Step 6: Commit the contract/docs-only slice with a message such as `docs: define disentanglenet matrix_vis bridge contract`.

### Task 2: Add basis-export entrypoint for Step 1

**Files:**
- Create: `scripts/disentangleNet/analysis/export_matrix_vis_basis.py`
- Modify: `scripts/disentangleNet/analysis/analyze_side_interpretability.py`
- Modify: `scripts/disentangleNet/analysis/analyze_checkpoint.py`
- Create: `scripts/disentangleNet/tests/test_matrix_vis_exports.py`

- [ ] Step 1: Add a dedicated exporter that loads a frozen `v31` checkpoint, resolves its basis bank, and writes one `basis_bank_x.npy` stack plus a `basis_manifest.json`.
- [ ] Step 2: Include enough metadata in the manifest to drive config generation later: checkpoint path, mode, region, basis count, matrix size, level boundaries, side-basis count, and exported point-layout name.
- [ ] Step 3: Reuse existing basis-loading and checkpoint-loading helpers where possible instead of duplicating model bootstrap logic.
- [ ] Step 4: Keep the exporter independent from k-fold or side-interpretability reports so it can run quickly on any accepted checkpoint.
- [ ] Step 5: Add a lightweight unit test or fixture-driven smoke test that validates manifest fields and exported stack shape for a synthetic basis bank.
- [ ] Step 6: Run `pytest scripts/disentangleNet/tests/test_matrix_vis_exports.py -v` if the repo test layout permits it; otherwise add a narrow script-level smoke harness and document the alternative.
- [ ] Step 7: Commit the basis-export slice with a message such as `feat: add disentanglenet basis export for matrix_vis`.

### Task 3: Generate and run Step 1 basis-wise `matrix_vis` configs

**Files:**
- Create: `scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py`
- Create: `scripts/matrix_vis/scripts/run_disentanglenet_basis_batch.py`
- Create: `scripts/matrix_vis/configs/real/disentanglenet/basis_compose_y_static_template.yaml`
- Modify: `scripts/matrix_vis/io/config.py`
- Modify: `scripts/matrix_vis/README.md`

- [ ] Step 1: Build a config generator that reads `basis_manifest.json` and emits one `x` reconstruction config per basis under `scripts/matrix_vis/configs/real/disentanglenet/generated/`.
- [ ] Step 2: Base each generated `x` config on the existing real-data conventions from [svd_pc1_axis_x_full341_anchor_facebox_matrixfree.yaml](/home/weizilin/generate_idea/scripts/matrix_vis/configs/real/svd_pc1_axis_x_full341_anchor_facebox_matrixfree.yaml:1), but switch the basis source to the exported `disentangleNet` stack and set `basis_index` explicitly.
- [ ] Step 3: Decide whether the generated configs should use the native `mouth` subset or a mapped full-layout contract; implement only one supported path in the first version and record the unsupported alternative in docs.
- [ ] Step 4: Generate a matching compose config for each basis that pairs its `x` solution with the fixed `y` solution from [svd_pc1_axis_y_full341_anchor_facebox_matrixfree.yaml](/home/weizilin/generate_idea/scripts/matrix_vis/configs/real/svd_pc1_axis_y_full341_anchor_facebox_matrixfree.yaml:1) or a cached run of that config.
- [ ] Step 5: Add a batch runner that executes `reconstruct_axis` for every generated `x` config and then `compose_motion` for every generated compose config, saving a summary table of output paths and solver diagnostics.
- [ ] Step 6: Add narrow config-loader coverage if `matrix_vis` requires any new fields for generated real configs.
- [ ] Step 7: Run `python scripts/matrix_vis/scripts/generate_disentanglenet_basis_configs.py --help` and one smoke generation against a real exported manifest.
- [ ] Step 8: Run one basis-only smoke reconstruction and one basis-only compose pass before attempting the full batch.
- [ ] Step 9: Commit the basis-config generation slice with a message such as `feat: add basis-wise matrix_vis bridge pipeline`.

### Task 4: Add patient inference export for `TT/844697`

**Files:**
- Create: `scripts/disentangleNet/analysis/export_matrix_vis_patient.py`
- Modify: `scripts/disentangleNet/analysis/analyze_side_interpretability.py`
- Modify: `scripts/disentangleNet/data/datasets.py`
- Create: `scripts/disentangleNet/tests/test_patient_matrix_vis_export.py`

- [ ] Step 1: Add a dedicated patient exporter that filters evaluation data down to `dataset=TT`, `subject=844697`, and `mode=x`, then runs inference without mixing in unrelated subjects.
- [ ] Step 2: Export patient-level metadata: group IDs, valid window indices, side-label predictions or logits, per-window free/shared coefficients, and any basis usage signals needed for interpretation.
- [ ] Step 3: Define the exact coefficient-composition rule for Step 2 and implement it in the exporter so the output includes one composed `x` observation matrix per valid window.
- [ ] Step 4: Save these artifacts in a stable patient bundle such as `patient_844697_x_sequence.npz`, `patient_844697_side_predictions.csv`, and `patient_844697_summary.json`.
- [ ] Step 5: Ensure the exporter preserves the original window order and carries enough provenance to map each composed matrix back to `window_idx` and `prev_window_idx`.
- [ ] Step 6: Add a focused test or fixture-based smoke check that verifies ordered window export and composed-matrix shape consistency.
- [ ] Step 7: Run the exporter once on the target checkpoint and inspect the emitted window count, coefficient tensor shapes, and side-prediction file.
- [ ] Step 8: Commit the patient-export slice with a message such as `feat: add patient sequence export for matrix_vis`.

### Task 5: Extend `matrix_vis` to accept externally supplied initial positions

**Files:**
- Modify: `scripts/matrix_vis/core/types.py`
- Modify: `scripts/matrix_vis/io/config.py`
- Modify: `scripts/matrix_vis/pipelines/reconstruct.py`
- Modify: `scripts/matrix_vis/io/save_results.py`
- Create: `scripts/matrix_vis/tests/test_initial_position_override.py`

- [ ] Step 1: Add an optional config field for external initial positions, scoped narrowly so it affects only the `x` single-axis reconstruction input and does not change solver math.
- [ ] Step 2: Update the reconstruction pipeline so it uses overridden initial positions when provided, otherwise it falls back to projected mesh coordinates exactly as today.
- [ ] Step 3: Save the effective initial positions into `solution.npz` and `summary.json` so rolling-window debugging is possible.
- [ ] Step 4: Add a targeted test showing that the override path changes `initial_positions` and leaves all other config validation intact.
- [ ] Step 5: Run `pytest scripts/matrix_vis/tests/test_initial_position_override.py -v`.
- [ ] Step 6: Commit the initial-position-override slice with a message such as `feat: allow matrix_vis initial position overrides`.

### Task 6: Build the Step 2 patient rolling-sequence pipeline

**Files:**
- Create: `scripts/matrix_vis/pipelines/patient_sequence.py`
- Create: `scripts/matrix_vis/cli/reconstruct_patient_sequence.py`
- Create: `scripts/matrix_vis/scripts/run_tt_844697_sequence.py`
- Modify: `scripts/matrix_vis/README.md`
- Modify: `scripts/matrix_vis/USAGE_GUIDE_CN.md`

- [ ] Step 1: Implement a patient-sequence pipeline that reads the exported `844697` patient bundle, iterates windows in order, and calls the existing axis-reconstruction machinery once per composed `x` observation matrix.
- [ ] Step 2: For the first window, pass the static projected mesh positions as initial positions; for each later window, pass the previous window’s reconstructed final-frame `x` coordinates as the next initial positions.
- [ ] Step 3: Persist per-window outputs in a deterministic directory layout such as `outputs/matrix_vis/patients/TT_844697/window_###/`.
- [ ] Step 4: Emit a sequence manifest summarizing which source window produced which reconstruction directory, side prediction, coefficient vector, and effective initial positions.
- [ ] Step 5: Add a dedicated CLI wrapper for this pipeline rather than overloading the generic `reconstruct_axis` CLI.
- [ ] Step 6: Add a minimal unit test or toy-sequence smoke test that validates window chaining and first-window static initialization.
- [ ] Step 7: Run one partial-sequence smoke pass on the first two valid windows before running the full patient sequence.
- [ ] Step 8: Commit the patient-sequence slice with a message such as `feat: add rolling patient sequence reconstruction pipeline`.

### Task 7: Add `y`-static composition and patient-level visualization

**Files:**
- Modify: `scripts/matrix_vis/pipelines/compose.py`
- Modify: `scripts/matrix_vis/io/compose_config.py`
- Create: `scripts/matrix_vis/pipelines/compose_patient_static_y.py`
- Create: `scripts/matrix_vis/cli/compose_patient_static_y.py`
- Create: `scripts/matrix_vis/tests/test_compose_patient_static_y.py`

- [ ] Step 1: Add a narrow composition mode for patient sequences where `x` comes from per-window or per-sequence reconstructions and `y` is held fixed at the mesh template coordinates.
- [ ] Step 2: Keep this mode separate from the existing generic `x+y` composition path so Step 1 basis-wise composition stays unchanged.
- [ ] Step 3: Save patient-level snapshots, frame directories, and a GIF or equivalent preview that makes the rolling `x` motion easy to inspect.
- [ ] Step 4: Record in `composed_summary.json` that the `y` channel is static and list the source patient export plus source window directories.
- [ ] Step 5: Add one test covering shape alignment and static-`y` coordinate filling.
- [ ] Step 6: Run `pytest scripts/matrix_vis/tests/test_compose_patient_static_y.py -v`.
- [ ] Step 7: Run one smoke composition on a short exported `844697` sequence before scaling to the full patient.
- [ ] Step 8: Commit the static-`y` patient visualization slice with a message such as `feat: add static-y patient motion composition`.

### Task 8: End-to-end validation and operator runbook

**Files:**
- Modify: `scripts/disentangleNet/README.md`
- Modify: `scripts/matrix_vis/README.md`
- Modify: `scripts/matrix_vis/USAGE_GUIDE_CN.md`
- Create: `docs/disentanglenet_matrix_vis_runbook.md`

- [ ] Step 1: Record the exact end-to-end commands for Step 1: export basis bank, generate basis configs, run one `x` reconstruction per basis, and compose each with the fixed `y` solution.
- [ ] Step 2: Record the exact end-to-end commands for Step 2: export `TT/844697`, run rolling `x` sequence reconstruction, and compose with static `y`.
- [ ] Step 3: Define the acceptance checklist for Step 1: all bases reconstruct without shape errors, all compose outputs exist, and diagnostics are written.
- [ ] Step 4: Define the acceptance checklist for Step 2: patient windows are ordered correctly, side predictions are exported, rolling initial positions change after window 1, and the final animation is rendered.
- [ ] Step 5: Run the smallest full-chain smoke path that touches both Step 1 and Step 2, capture output directories, and add them to the runbook as known-good examples.
- [ ] Step 6: Commit the validation/docs slice with a message such as `docs: add disentanglenet matrix_vis runbook`.

