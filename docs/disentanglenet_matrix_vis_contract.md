# DisentangleNet to Matrix Vis Bridge Contract

## Scope

This contract defines the first supported bridge between `scripts/disentangleNet`
and `scripts/matrix_vis`.

Two workflows are in scope:

1. Step 1: basis-wise post-hoc analysis
   - Train or reuse a frozen `disentangleNet` `v31` checkpoint in `mode=x`,
     `region=mouth`.
   - Export the learned `x` basis stack.
   - Reconstruct one `x` trajectory per basis with `matrix_vis`.
   - Compose each `x` result with one fixed `y` reconstruction produced from
     the existing SVD config.

2. Step 2: patient-wise coefficient-composed post-hoc analysis
   - Run inference on one target patient, currently `TT/844697`.
   - Export patient-specific coefficients, usage, side predictions, and one
     composed `x` observation matrix per valid window.
   - Reconstruct the patient sequence in `matrix_vis` with rolling
     `x` initial positions.
   - Keep `y` static in the first implementation.

Out of scope for this bridge:

- Directly solving on `disentangleNet` latent vectors inside `matrix_vis`
- Joint `x+y` coefficient composition inside the same inverse problem
- Full-face expansion from `mouth` observations to full341 observations
- Generic multi-patient orchestration beyond the first validated path

## Artifact Root

Bridge artifacts must be written under:

`outputs/disentangleNet/<run_name>/matrix_vis_exports/`

Recommended layout:

```text
matrix_vis_exports/
  basis/
    basis_bank_x.npy
    basis_manifest.json
  patients/
    TT_844697/
      patient_844697_x_sequence.npz
      patient_844697_side_predictions.csv
      patient_844697_summary.json
```

## Step 1: Basis-wise Export Contract

### `basis_bank_x.npy`

- dtype: `float32`
- shape: `[K, 119, 119]`
- semantics: one signed `mean_distance_delta` matrix per learned `x/mouth`
  basis
- ordering: identical to the exported `basis_manifest.json`

### `basis_manifest.json`

Required fields:

- `checkpoint_path`
- `mode`
- `region`
- `matrix_size`
- `num_basis`
- `levels`
- `level_boundaries`
- `basis_orthogonalization`
- `quantizer_type`
- `point_layout`
- `value_semantics`
- `exported_basis_path`

Current fixed assumptions:

- `mode = x`
- `region = mouth`
- `matrix_size = 119`
- `point_layout = face_regions_grouped`
- `point_layout_region_names = [around_mouth, mouth]`
- `value_semantics = mean_distance_delta`

`matrix_vis` treats each exported basis matrix as one single-axis observation
matrix. It does not interpret these matrices as latent factors.

## Step 2: Patient Export Contract

Current target:

- dataset: `TT`
- subject: `844697`
- mode: `x`

### `patient_844697_x_sequence.npz`

Required arrays:

- `window_indices`
  - shape: `[W]`
  - strictly increasing valid windows used for sequence reconstruction
- `prev_window_indices`
  - shape: `[W]`
  - source previous window index per valid window
- `side_pred`
  - shape: `[W]`
  - predicted side label or group-level side decision aligned with each window
- `side_logits`
  - optional, shape: `[W, C]`
- `basis_coeffs`
  - shape: `[W, K]`
  - per-window coefficients used to compose the `x` observation matrices
- `basis_usage`
  - optional, shape: `[W, K]`
  - per-window usage or contribution signal for interpretation
- `composed_basis_matrices`
  - shape: `[W, 119, 119]`
  - coefficient-composed signed `mean_distance_delta` matrices

### `patient_844697_side_predictions.csv`

One row per exported window with:

- `dataset_name`
- `subject`
- `window_idx`
- `prev_window_idx`
- `side_pred`
- `side_label_name`
- any coefficient summary columns needed for quick inspection

### `patient_844697_summary.json`

Required fields:

- `checkpoint_path`
- `dataset_name`
- `subject`
- `num_valid_windows`
- `mode`
- `region`
- `point_layout`
- `matrix_size`
- `composition_rule`

## Composition Rule

Step 2 does not reconstruct per-basis trajectories independently.

Instead, for each exported valid patient window:

1. Read the per-window coefficient vector.
2. Linearly compose one `x` observation matrix from the exported basis bank.
3. Feed that single composed matrix to `matrix_vis`.

The exact coefficient source must be recorded in `composition_rule`, for example:

- `shared_basis_coefficients`
- `side_path_usage_weighted`
- another explicit future rule

The exporter must not leave this ambiguous.

## Point Ordering

Current first implementation assumes:

- `disentangleNet` export order equals the grouped face-region layout built from
  `around_mouth` followed by `mouth`
- `matrix_vis` config uses `face_regions_grouped` with
  `region_names=[around_mouth, mouth]`

If full341 mapping is introduced later, the bridge must add an explicit layout
mapping artifact instead of silently reusing the cropped order.

## Patient Sequence Rule

Step 2 uses the following rolling initialization:

1. First exported window starts from the static projected mesh positions.
2. Window `t+1` starts from the reconstructed final-frame `x` positions of
   window `t`.
3. `y` remains static in the first implementation.

This rule must be visible in the sequence manifest and in the output summary.

## Fixed `y` Source for Step 1

The current fixed `y` reference is the existing real-data config:

`scripts/matrix_vis/configs/real/svd_pc1_axis_y_full341_anchor_facebox_matrixfree.yaml`

Step 1 composition pairs:

- one basis-wise `x` reconstruction from `disentangleNet`
- one fixed `y` reconstruction from the config above

## Operator Checklist

Before running the bridge:

1. Confirm the checkpoint is a frozen `disentangleNet v31` `x/mouth` run.
2. Export basis artifacts under `matrix_vis_exports/basis/`.
3. Run or reuse the fixed `y` reconstruction once.
4. For Step 1, generate one `x` config per basis and compose each with the same
   `y` solution.
5. For Step 2, export `TT/844697`, verify ordered valid windows, then run the
   rolling `x` sequence reconstruction with static `y`.
