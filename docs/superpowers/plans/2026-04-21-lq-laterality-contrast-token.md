# LQ Laterality Contrast Token Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit laterality-contrast side readout on top of the `v28` early-branch baseline and validate that the new side pooling mode runs correctly on the `win20 IMR+TT` dataset.

**Architecture:** Keep the `v28` shared trunk, free branch, private branch, and residual FSQ stack unchanged. Only the early side path changes: pool four fixed region tokens from the `15x15` side feature map, collapse them into two signed contrast tokens (`around-left - around-right`, `mouth-left - mouth-right`), then feed the concatenated `64`-dim readout into the existing side head and side basis path.

**Tech Stack:** Python, PyTorch, Fire CLI, Bash, existing `scripts/lq` train/analyze workflow, `dl` conda environment

---

## File Map

- Modify: `scripts/lq/model/distnet.py`
  - Add a new explicit laterality-contrast side pooling mode for the early side path and wire the side head input dim accordingly.
- Create: `scripts/lq/run_train_x_mouth_v29_laterality_contrast_probe.sh`
  - Clone `v28`, switch the side pooling mode to the new contrast-token readout, and write outputs to a canonical `v29` directory.
- Optional modify: `RESEARCH_PROGRESS.md`
  - Only after smoke or full run results exist.

## Constraints

- Every Python command must run under `dl`:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python ...
```

- Keep `side_basis_count = 4`
- Keep `side_z_dim = 32`
- Do not change the shared trunk, free/private branch, quantizer, or loss definitions in this round
- New behavior must be selected by explicit `side_pooling` config

## Task 1: Add The Laterality Contrast Pooling Mode

**Files:**
- Modify: `scripts/lq/model/distnet.py`
- Test: ad-hoc shape smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing shape smoke**

```python
import torch

from scripts.lq.model.distnet import DistNet

model = DistNet(
    levels=(2, 3, 6),
    basis_size=119,
    hidden_dim=32,
    private_dim=32,
    quantizer_type="residual_fsq",
    side_semantic_enabled=True,
    side_basis_count=4,
    side_pooling="fixed_region2_contrast",
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
    private_pool_size=1,
    side_z_dim=32,
)

side_feats = torch.randn(5, 32, 15, 15)
pooled = model._pool_side_tokens_region_contrast(side_feats)

assert pooled.shape == (5, 64)
print("side-contrast-pool-shape-ok")
```

- [ ] **Step 2: Run it and verify it fails before implementation**

- [ ] **Step 3: Add an explicit signed-token correctness smoke**

Use a synthetic `side_feats` tensor where only one region is filled, and verify:

- activating only `around_left` makes the first 32 dims positive and the second 32 dims zero
- activating only `around_right` makes the first 32 dims negative and the second 32 dims zero
- activating only `mouth_left` makes the second 32 dims positive and the first 32 dims zero
- activating only `mouth_right` makes the second 32 dims negative and the first 32 dims zero

- [ ] **Step 4: Implement the minimal helper and constructor wiring**

Requirements:

- Pool the same four fixed diagonal blocks used by `v28`
- Build two signed contrast tokens:
  - `around_left - around_right`
  - `mouth_left - mouth_right`
- Concatenate them to `[N, 64]`
- Set `self.side_pooled_dim = hidden_dim * 2` when `side_pooling == "fixed_region2_contrast"`
- In `_forward_early_branch(...)`, branch explicitly on the new mode

- [ ] **Step 5: Re-run the shape smoke**

Expected: prints `side-contrast-pool-shape-ok`

## Task 2: Verify Grouped Forward Contracts

**Files:**
- Modify: `scripts/lq/model/distnet.py`
- Test: ad-hoc grouped forward smoke

- [ ] **Step 1: Run a grouped forward smoke with `side_pooling=fixed_region2_contrast`**

Check:

- `side_path_usage.shape == (B, T, 4)`
- `side_path_representation.shape == (B, T, 4)`
- `group_pooled_side_rep.shape == (B, 32)`

- [ ] **Step 2: Verify no backward-compatibility regressions for old pooling modes**

Minimum compatibility check:

- instantiate `side_pooling=fixed_block4_diag`
- verify `_pool_side_tokens_fixed_blocks(torch.randn(N, 32, 15, 15)).shape == (N, 128)`

## Task 3: Add The Canonical `v29` Training Preset

**Files:**
- Create: `scripts/lq/run_train_x_mouth_v29_laterality_contrast_probe.sh`

- [ ] **Step 1: Clone the `v28` preset**

- [ ] **Step 2: Change only the intended deltas**

Required deltas:

- `--side_pooling=fixed_region2_contrast`
- `--output_dir=outputs/lq_x_mouth_v29_laterality_contrast_probe_win20_e50`

- [ ] **Step 3: Run `bash -n` on the new preset**

## Task 4: Run A Short Training Smoke

**Files:**
- Use: `scripts/lq/train.py`
- Use: `scripts/lq/analyze_checkpoint.py`

- [ ] **Step 1: Run a short `3` epoch smoke on `data/win20-step20/IMR,data/win20-step20/TT`**

Use the `v29` config but override:

- `--epochs=3`
- `--output_dir=outputs/lq_x_mouth_v29_laterality_contrast_probe_smoke`

- [ ] **Step 2: Run checkpoint analysis on the smoke best checkpoint**

- [ ] **Step 3: Record whether the run is structurally sane**

Minimum sanity checks:

- training completes without shape/runtime errors
- checkpoint analysis completes
- `side_from_side_rep_acc > side_from_free_rep_acc`
- no obvious loss explosion

## Task 5: Run The Canonical 50 Epoch Probe And Compare Against Baselines

**Files:**
- Use: `scripts/lq/run_train_x_mouth_v29_laterality_contrast_probe.sh`
- Use: `scripts/lq/analyze_checkpoint.py`
- Optional modify: `RESEARCH_PROGRESS.md`

- [ ] **Step 1: Run the full `v29` 50 epoch training preset**

- [ ] **Step 2: Analyze the final `best.pt` checkpoint**

Output dir:

- `outputs/lq_x_mouth_v29_laterality_contrast_probe_win20_e50/analysis`

- [ ] **Step 3: Compare against the pinned `v26` baseline and the `v28` side-aware pooling run**

Minimum comparison fields:

- `val_loss`
- `val_recon`
- `val_side_group`
- `side_from_side_rep_acc`
- `side_from_free_rep_acc`
- `dataset_from_side_rep_acc`
- `dataset_from_free_rep_acc`
- `dataset_from_private_rep_acc`
- `ortho_linear_r2_free_to_side`
- `ortho_linear_r2_side_to_free`

- [ ] **Step 4: Decide whether explicit laterality contrast is a better baseline than `v28`**

Primary success signal:

- `v29 side_from_side_rep_acc > v28`
- `v29 dataset_from_side_rep_acc < v28`

- [ ] **Step 5: Update `RESEARCH_PROGRESS.md` with the result and next-step recommendation**
