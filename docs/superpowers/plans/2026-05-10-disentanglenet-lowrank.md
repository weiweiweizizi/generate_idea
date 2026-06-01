# DisentangleNet Lowrank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `scripts/disentangleNet_lowrank` experiment that hard-codes low-rank, symmetric, zero-diagonal, mutually orthogonal action bases while retaining DCT smoothness regularization.

**Architecture:** Reuse the `scripts/disentangleNet` V6/V9 training path and replace the directly learned action basis bank with a grouped global orthogonal vector pool. Each basis is assembled from a disjoint group of orthonormal vectors, giving strict rank caps and Frobenius-orthogonal bases by construction.

**Tech Stack:** Python, PyTorch, existing `scripts.disentangleNet` data/training utilities.

---

### Task 1: Low-Rank Basis Parameterization

**Files:**
- Create: `scripts/disentangleNet_lowrank/model/lowrank_basis.py`

- [ ] Implement SVD-based factor initialization from existing `basis_x_shared_2_6.npy` / `basis_y_shared_2_6.npy`.
- [ ] Implement grouped global QR over latent vectors.
- [ ] Implement basis synthesis as `B = U @ diag(scale) @ U.T`, then zero the diagonal.
- [ ] Add diagnostics for rank, symmetry, diagonal magnitude, and inter-basis Gram error.

### Task 2: Model Wrapper

**Files:**
- Create: `scripts/disentangleNet_lowrank/model/lowrank_distnet.py`
- Create: `scripts/disentangleNet_lowrank/model/__init__.py`
- Create: `scripts/disentangleNet_lowrank/__init__.py`

- [ ] Subclass `scripts.disentangleNet.model.v6_distnet.V6DistNet`.
- [ ] Disable direct `action_basis_bank` learning.
- [ ] Override `get_structured_basis()` to use low-rank synthesis.
- [ ] Return `lowrank_freq_loss` and diagnostics in forward outputs.

### Task 3: Loss And Training Entrypoint

**Files:**
- Create: `scripts/disentangleNet_lowrank/lowrank_loss.py`
- Create: `scripts/disentangleNet_lowrank/train.py`

- [ ] Build loss weights with `lq=0`, `orth=0`, and DCT frequency regularization retained.
- [ ] Reuse existing `scripts.disentangleNet.training.run_epoch`.
- [ ] Save outputs under `outputs/disentangleNet_lowrank/`.

### Task 4: Smoke Test

**Files:**
- Create: `scripts/disentangleNet_lowrank/tests/test_lowrank_smoke.py`

- [ ] Instantiate `LowRankDistNet`.
- [ ] Verify forward/backward on random `[B,T,1,119,119]`.
- [ ] Verify basis symmetry, zero diagonal, rank caps, and off-diagonal Gram near zero.
- [ ] Run `python scripts/disentangleNet_lowrank/tests/test_lowrank_smoke.py`.
