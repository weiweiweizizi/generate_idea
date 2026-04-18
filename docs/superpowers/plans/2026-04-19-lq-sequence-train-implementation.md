# LQ Sequence Train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/lq/train.py` train the single-direction LQ model on grouped sequence samples with basis initialization and a batch-memory smoke test.

**Architecture:** Keep `FacialMotionSequenceDataset` as the source of grouped samples and extend `DistNet` with a thin sequence adapter that flattens `B x T` into `B*T`, reuses the existing frame-wise LQ path, then reshapes outputs back to sequence form. Compute all train-time losses in `train.py` with explicit masks so `valid`, `deleted_fill`, and `zero_pad` windows contribute differently.

**Tech Stack:** Python, PyTorch, Fire CLI, existing `scripts/lq` modules

---

### Task 1: Sequence-Capable `DistNet`

**Files:**
- Modify: `scripts/lq/model/network.py`

- [ ] **Step 1: Add sequence flatten/unflatten helpers**

Add small internal helpers that:
- accept `B x 1 x H x W` or `B x T x 1 x H x W`
- flatten sequence inputs to `B*T x 1 x H x W`
- record whether time dimension exists and the original `(B, T)` shape
- reshape frame-wise outputs back to `B x T x ...` when needed

- [ ] **Step 2: Keep the current frame-wise forward path intact**

Refactor the existing forward body into a frame-wise path that still operates on `N x 1 x H x W` tensors so the LQ / basis / residual logic stays unchanged.

- [ ] **Step 3: Return per-frame auxiliary losses**

Expose unreduced frame-wise values needed by masked training:
- `lq_loss_per_sample`
- `residual_l1_per_sample`
- `side_loss_cont_per_sample`
- `side_loss_disc_per_sample`
- `side_loss_per_sample`
- `dataset_private_loss_per_sample`
- `dataset_adv_loss_per_sample`

Keep legacy scalar aliases:
- `lq_loss`
- `residual_l1`
- `side_loss["side_loss"]`
- `dataset_loss[...]`

- [ ] **Step 4: Reshape all sequence-aware outputs**

For sequence input, reshape:
- reconstructions
- latents
- indices
- decoded indices
- logits
- per-sample losses

Leave `orth_loss`, `action_basis`, and `basis` as batch-global values.

---

### Task 2: Sequence Dataset Training Path

**Files:**
- Modify: `scripts/lq/train.py`

- [ ] **Step 1: Switch dataset construction to grouped sequences**

Replace `FacialMotionDataset` with `FacialMotionSequenceDataset` in `build_datasets()` and thread through:
- `group_size`
- `apply_deleted_filter`

Use grouped dataset `compute_global_scale()` when `signed_normalize="global"`.

- [ ] **Step 2: Add masked-loss helpers**

Implement small helpers in `train.py` for:
- estimating tensor memory in MiB
- masked mean on `B x T` tensors
- expanding sequence labels from `B` to `B x T` when needed

- [ ] **Step 3: Rewrite `step_model()` for sequence batches**

Read:
- `images`
- `valid_mask`
- `padding_mask`
- `side_label`
- `dataset_label`

Compute:
- `recon_mask = ~padding_mask`
- `supervision_mask = valid_mask`

Use model outputs to compute masked losses:
- `recon` on `recon_mask`
- `lq` on `recon_mask`
- `residual` on `recon_mask`
- `side` on `supervision_mask`
- `dataset_private` on `supervision_mask`
- `dataset_adv` on `supervision_mask`
- `orth` once per batch

- [ ] **Step 4: Save richer checkpoints**

Include in `best.pt`:
- model state
- epoch
- train metrics
- val metrics
- config snapshot including `mode`, `region`, `levels`, `group_size`, and `action_basis_init_path`

---

### Task 3: Batch Memory Smoke Test

**Files:**
- Modify: `scripts/lq/train.py`

- [ ] **Step 1: Add CLI options for first-pass experiments**

Add or update defaults for:
- `batch_size=64`
- `group_size=4`
- `validate_batch_memory=True`
- `require_basis_init=True`

- [ ] **Step 2: Implement one-batch memory validation**

Before the training loop:
- fetch a single batch from the train loader
- print `images.shape`, mask shapes, and estimated input memory
- run one forward/backward smoke pass
- zero gradients after the smoke pass

- [ ] **Step 3: Make basis init explicit**

If `require_basis_init=True` and `action_basis_init_path` is missing, raise a clear error. If disabled, print a warning before training starts.

---

### Task 4: Validation

**Files:**
- Test via commands only

- [ ] **Step 1: Verify syntax**

Run:

```bash
python -m py_compile scripts/lq/train.py scripts/lq/model/network.py scripts/lq/datasets.py
```

Expected: no output, exit code 0

- [ ] **Step 2: Run import/smoke validation**

Run a small Python snippet that:
- imports `DistNet`
- creates fake `B x T x 1 x H x W` input
- verifies output shapes

Expected: prints sequence output shapes without exceptions

- [ ] **Step 3: If data is locally available, run a one-batch train smoke**

Run:

```bash
python scripts/lq/train.py --epochs=1 --batch_size=64 --group_size=4 --validate_batch_memory=True --mode=x --region=mouth --action_basis_init_path=<path>
```

Expected:
- batch memory info printed
- one batch forward/backward smoke succeeds
- training enters epoch loop without shape or mask errors
