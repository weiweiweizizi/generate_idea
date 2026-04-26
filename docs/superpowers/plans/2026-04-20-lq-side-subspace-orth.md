# LQ Side Subspace Orthogonalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the shared quantized latent into dedicated side/free subspaces and add a batch-level orthogonality penalty between them.

**Architecture:** Keep the current residual-FSQ shared encoder/quantizer unchanged, but hard-split `shared_quantized` into `z_side` and `z_free` after quantization. Route side-specific heads through `z_side`, route free-path heads through `z_free`, and compute a masked cross-correlation penalty over valid frames during loss assembly. Extend checkpoint analysis to expose the new latent subspaces for post-hoc probes.

**Tech Stack:** Python, PyTorch, Fire, existing `scripts/lq` training and analysis pipeline

---

### Task 1: Add Subspace Config Surface

**Files:**
- Modify: `scripts/lq/train.py`
- Modify: `scripts/lq/training/config.py`

- [ ] Add `side_subspace_dim` and `subspace_orth_weight` CLI/config fields.
- [ ] Validate that `0 < side_subspace_dim < shared_dim` whenever `side_semantic_enabled=True`.
- [ ] Thread the new fields from `train.py` into `DistNet` construction and `loss_weights`.

### Task 2: Split Shared Quantized Latent In DistNet

**Files:**
- Modify: `scripts/lq/model/distnet.py`
- Modify: `scripts/lq/model/heads.py`

- [ ] Create explicit `self.side_subspace_dim` / `self.free_subspace_dim`.
- [ ] Build side heads from `side_subspace_dim` and free-path heads from `free_subspace_dim`.
- [ ] Split `shared_quantized` into `side_latent` and `free_latent` inside `forward()`.
- [ ] Route side semantic heads only through `side_latent`.
- [ ] Route free-path basis/coeff heads only through `free_latent`.
- [ ] Expose `side_latent` and `free_latent` in model outputs for loss/analysis.

### Task 3: Add Masked Subspace Orthogonality Loss

**Files:**
- Modify: `scripts/lq/training/losses.py`

- [ ] Implement a masked batch-level cross-correlation penalty between `z_side` and `z_free`.
- [ ] Add the weighted penalty into total loss as `subspace_orth`.
- [ ] Report the raw metric in epoch logs.

### Task 4: Extend Analysis For New Latent Subspaces

**Files:**
- Modify: `scripts/lq/analyze_checkpoint.py`

- [ ] Save pooled `side_latent` / `free_latent` group representations.
- [ ] Include summary statistics needed to compare side/free latent leakage later.
- [ ] Keep analysis backward-compatible with older checkpoints.

### Task 5: Smoke Validation

**Files:**
- Modify: `scripts/lq/run_train_x_mouth_v22_side_semantic_bank_probe.sh` if a new preset is needed, otherwise run inline CLI.

- [ ] Run a 1-epoch smoke test under `conda activate dl`.
- [ ] Run checkpoint analysis on the smoke checkpoint.
- [ ] Check that training is stable and that `subspace_orth` is finite.
- [ ] Compare side/free probe directionally and inspect whether free-to-side linear recoverability drops.
