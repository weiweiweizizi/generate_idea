# DisentangleNet V31 Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the accepted `v31` training and probe-analysis stack into a self-contained `scripts/disentangleNet` package with redundant compatibility layers removed.

**Architecture:** Copy only the `v31` runtime closure from `scripts/lq` into a new package rooted at `scripts/disentangleNet`, then rewrite imports so training and analysis no longer depend on `scripts/lq`. Keep the post-hoc probe path available under `analysis/`, but trim shims and unrelated experiment entrypoints.

**Tech Stack:** Python, PyTorch, NumPy, pandas, Fire, Pillow, tqdm, optional scikit-learn, optional vector-quantize-pytorch/einops

---

### Task 1: Define the extracted package boundary

**Files:**
- Create: `docs/superpowers/plans/2026-04-25-disentanglenet-v31-freeze.md`
- Modify: `scripts/disentangleNet/README.md`

- [ ] Step 1: Enumerate the `v31` runtime closure from `train.py`, `training/`, `data/`, `model/`, `init_basis/`, and probe analyzers.
- [ ] Step 2: Exclude unrelated experiment wrappers and compatibility shims such as `scripts/lq/model/network.py` and `scripts/lq/datasets.py`.
- [ ] Step 3: Document the final directory layout and purpose of each subdirectory in `README.md`.

### Task 2: Extract the self-contained training package

**Files:**
- Create: `scripts/disentangleNet/train.py`
- Create: `scripts/disentangleNet/regions.py`
- Create: `scripts/disentangleNet/data/__init__.py`
- Create: `scripts/disentangleNet/data/datasets.py`
- Create: `scripts/disentangleNet/data/io.py`
- Create: `scripts/disentangleNet/data/samples.py`
- Create: `scripts/disentangleNet/data/specs.py`
- Create: `scripts/disentangleNet/model/__init__.py`
- Create: `scripts/disentangleNet/model/BasicBlock.py`
- Create: `scripts/disentangleNet/model/basis.py`
- Create: `scripts/disentangleNet/model/distnet.py`
- Create: `scripts/disentangleNet/model/encoder.py`
- Create: `scripts/disentangleNet/model/heads.py`
- Create: `scripts/disentangleNet/model/latent_quantization.py`
- Create: `scripts/disentangleNet/model/quantizers.py`
- Create: `scripts/disentangleNet/training/__init__.py`
- Create: `scripts/disentangleNet/training/checkpoint.py`
- Create: `scripts/disentangleNet/training/config.py`
- Create: `scripts/disentangleNet/training/data.py`
- Create: `scripts/disentangleNet/training/engine.py`
- Create: `scripts/disentangleNet/training/losses.py`
- Create: `scripts/disentangleNet/init_basis/basis_x_shared_2_6.npy`
- Create: `scripts/disentangleNet/init_basis/basis_x_side_from_level2.npy`

- [ ] Step 1: Copy the minimal training/data/model files used by `v31`.
- [ ] Step 2: Rewrite imports to target `scripts.disentangleNet` directly.
- [ ] Step 3: Remove fallback import branches that only exist for old `scripts/lq` layouts.
- [ ] Step 4: Keep behavior unchanged for `v31` training arguments and checkpoint format.

### Task 3: Preserve the probe analysis path

**Files:**
- Create: `scripts/disentangleNet/analysis/analyze_checkpoint.py`
- Create: `scripts/disentangleNet/analysis/analyze_side_interpretability.py`
- Create: `scripts/disentangleNet/analysis/analyze_kfold_report.py`

- [ ] Step 1: Copy the three analysis entrypoints required for post-hoc probe workflows.
- [ ] Step 2: Retarget them to `scripts.disentangleNet` imports.
- [ ] Step 3: Trim references to the old package and remove dead compatibility code where safe.

### Task 4: Add the final v31 entrypoints and documentation

**Files:**
- Create: `scripts/disentangleNet/run_train_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe.sh`
- Create: `scripts/disentangleNet/README.md`

- [ ] Step 1: Point the run script at `scripts/disentangleNet/train.py`.
- [ ] Step 2: Point basis init paths at `scripts/disentangleNet/init_basis/...`.
- [ ] Step 3: Move outputs to `outputs/disentangleNet/...` to avoid mixing with `scripts/lq`.
- [ ] Step 4: Document the preserved `v31` hyperparameters and available analysis commands.

### Task 5: Smoke-check the extraction

**Files:**
- Test: `scripts/disentangleNet/train.py`
- Test: `scripts/disentangleNet/analysis/analyze_checkpoint.py`
- Test: `scripts/disentangleNet/analysis/analyze_side_interpretability.py`
- Test: `scripts/disentangleNet/analysis/analyze_kfold_report.py`

- [ ] Step 1: Run import-only smoke checks on train and analysis entrypoints.
- [ ] Step 2: Run CLI help or module parse checks to catch broken imports.
- [ ] Step 3: Summarize what was kept, what was cut, and any residual runtime dependencies.
