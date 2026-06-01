# Strict OOF DisentangleNet Trainprobe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict k-fold OOF training and analysis support to `scripts/disentangleNet_trainprobe` and its downstream patient-level export chain.

**Architecture:** Introduce deterministic subject-fold manifests in the training/data layer, write one checkpoint per fold, then export OOF window activations by running each fold checkpoint only on that fold's validation subjects. Reuse existing patient-summary and t-SNE code by feeding them merged OOF outputs instead of a full-data checkpoint export.

**Tech Stack:** Python, PyTorch, pandas, NumPy, Fire

---

### Task 1: Add fold assignment utilities

**Files:**
- Modify: `scripts/disentangleNet_trainprobe/training/data.py`
- Modify: `scripts/disentangleNet_trainprobe/analysis/common.py`

- [ ] Add deterministic helpers to build per-spec subject folds.
- [ ] Add dataset builders that accept explicit train/val subject lists.
- [ ] Add manifest-friendly serialization helpers for folds.

### Task 2: Add k-fold training entrypoints

**Files:**
- Modify: `scripts/disentangleNet_trainprobe/train.py`

- [ ] Extend config to include `num_folds`, `fold_index`, and fold output metadata.
- [ ] Support single-fold training with explicit fold selection.
- [ ] Support all-fold execution that loops through folds and writes a root `fold_manifest.json`.
- [ ] Save per-epoch metric history for each fold.

### Task 3: Add strict OOF export path

**Files:**
- Modify: `scripts/disentangleNet_trainprobe/analysis/export_window_basis_activations.py`
- Modify: `scripts/disentangleNet_trainprobe/analysis/common.py`

- [ ] Add fold-manifest-driven export mode.
- [ ] Export validation-only windows per fold.
- [ ] Merge per-fold CSVs into one strict OOF output directory with summary metadata.

### Task 4: Wire patient summary and t-SNE to OOF inputs

**Files:**
- Modify: `scripts/disentangleNet/analysis/analyze_patient_tsne.py`
- Optionally modify: `scripts/disentangleNet/analysis/README.md`

- [ ] Deprecate the current pseudo-OOF fold-annotation path.
- [ ] Keep t-SNE working from explicit patient profile CSV inputs.
- [ ] Update docs/comments so strict OOF means “generated from OOF patient profiles”.

### Task 5: Verify end-to-end behavior

**Files:**
- No new source files required unless a small helper script is needed

- [ ] Run lightweight validation on fold manifests and subject uniqueness.
- [ ] Run one small k-fold training/extraction smoke path if feasible.
- [ ] Re-run OOF export checks and confirm merged outputs are validation-only.

