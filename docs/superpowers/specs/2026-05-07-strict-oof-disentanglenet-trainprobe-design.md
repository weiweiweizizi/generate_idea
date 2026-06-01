# Strict OOF DisentangleNet Trainprobe Design

## Goal

Implement a strict out-of-fold workflow for `scripts/disentangleNet_trainprobe` so that:

- training can run across `k` subject folds,
- each fold writes its own checkpoint and subject split metadata,
- OOF window-level exports are produced only from each fold's validation subjects,
- downstream patient summaries and t-SNE inputs are assembled from true OOF artifacts instead of a single full-data checkpoint.

## Current Problem

The current repository trains a single train/val split using `subject_split(spec, val_ratio, seed)`. Downstream analysis then reuses one checkpoint and, in the t-SNE path, can only annotate patients with fold ids after the fact. That is not strict OOF because the exported patient features do not come from per-fold validation-only model outputs.

## Scope

This change covers:

- `scripts/disentangleNet_trainprobe` training data split generation
- train CLI support for per-fold and all-fold execution
- OOF window activation export from fold checkpoints
- OOF patient profile assembly
- OOF t-SNE input plumbing

This change does not attempt to redesign the existing disentangleNet mainline training code outside `trainprobe`.

## Design

### 1. Fold manifests and subject splits

Introduce deterministic subject-fold assignment at the dataset-spec level. Each subject belongs to exactly one fold. The assignment must be saved so downstream analysis can consume the exact same splits used for training.

Artifacts:

- root-level `fold_manifest.json`
- per-fold `subject_split.json`

### 2. K-fold training outputs

Training supports:

- single-fold training for `fold_index=i`
- all-fold training for `num_folds=k`

Each fold writes to:

- `output_root/fold_{i}/best.pt`
- `output_root/fold_{i}/subject_split.json`
- `output_root/fold_{i}/metrics_history.jsonl`

The root output directory stores the global fold manifest and a summary of all folds.

### 3. Strict OOF export

OOF export reads the fold manifest and loops over fold checkpoints. For each fold:

- load `fold_{i}/best.pt`
- construct only that fold's validation subject dataset
- export only those windows

The exporter writes both per-fold outputs and merged OOF outputs:

- `window_basis_activations_oof/per_fold/fold_{i}/...`
- `window_basis_activations_oof/window_basis_activations_wide.csv`
- `window_basis_activations_oof/window_basis_activations_long.csv`

### 4. Downstream patient summary and t-SNE

Patient summary continues to aggregate from a wide CSV, but now the input can be the strict OOF wide table.

t-SNE should no longer pretend to build OOF by attaching fold ids to a full patient table. Instead, OOF t-SNE is defined by pointing the script at the patient profiles generated from the strict OOF wide table.

The old fold-annotation branch should be marked deprecated or removed from the OOF narrative to avoid misuse.

## Validation

Required checks:

- every subject appears in exactly one fold
- no overlap between train and val subjects inside a fold
- merged OOF wide CSV has no duplicate `(dataset_name, subject, window_idx)` rows
- merged OOF patient table has one row per subject
- t-SNE summaries reference OOF patient profiles directly rather than synthetic fold annotations

