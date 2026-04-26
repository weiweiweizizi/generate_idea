# LQ Side-Aware Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `v28` early-branch probe that keeps the shared trunk fixed, replaces the side branch `2x2` pooling with fixed 4-block side-aware pooling, expands `side_basis_count` from `2` to `4`, and compares the result against the pinned `v26` reanalyzed baseline.

**Architecture:** Keep the `v26` early-branch free/private paths unchanged. Only the side readout changes: `side_adapter(feats)` will be pooled into four fixed diagonal block tokens on the `15x15` feature map, concatenated to `128` dims, then passed through the existing side head. The run is treated explicitly as a combined `side readout + side basis capacity` probe, not a pure pooling ablation.

**Tech Stack:** Python, PyTorch, `vector-quantize-pytorch`, Fire CLI, Bash, existing `scripts/lq` train/analyze workflow, `dl` conda environment

---

## File Map

- Modify: `scripts/lq/model/distnet.py`
  - Add fixed 4-block side-aware pooling for the early-branch side path, selected by an explicit `side_pooling` mode, keep free/private behavior unchanged, and preserve existing output contracts.
- Modify: `scripts/lq/analyze_checkpoint.py`
  - Reconstruct `DistNet` with the checkpoint’s explicit `side_pooling` mode so `v28` checkpoints remain reproducible and analyzable.
- Create: `scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh`
  - Clone the `v26` preset, switch to the explicit side-aware pooling config, restore `side_z_dim=32`, set `side_basis_count=4`, and write to the canonical `v28` output dir.
- Modify: `RESEARCH_PROGRESS.md`
  - Record the `v28` result against the pinned `v26 reanalyzed` baseline and state whether to escalate to the fallback laterality-contrast variant.

## Pinned References

- Approved spec:
  - `docs/superpowers/specs/2026-04-20-lq-side-aware-pooling-design.md`
- Pinned baseline checkpoint:
  - `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/best.pt`
- Pinned baseline analysis:
  - `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/summary.json`
- Baseline comparison values:
  - `side_from_side_rep_acc = 0.8227`
  - `side_from_free_rep_acc = 0.3273`
  - `dataset_from_side_rep_acc = 0.8182`
  - `dataset_from_free_rep_acc = 0.8091`
  - `dataset_from_private_rep_acc = 0.8909`
  - `val_loss = 0.7623`
  - `val_recon = 0.2873`
  - `val_side_group = 0.5365`
- Existing reference preset:
  - `scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh`
- New canonical candidate preset:
  - `scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh`
- New canonical candidate output:
  - `outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50`
- Dataset for all comparisons:
  - `data/win20-step20/IMR,data/win20-step20/TT`

## Constraints

- Every Python command must run under `dl`:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python ...
```

- Do not change the shared trunk, free path, quantizer stack, private path, or canonical analysis rule in this round.
- Treat this as a combined probe:
  - `side-aware pooling`
  - `side_basis_count = 2 -> 4`
- The new side-readout behavior must be checkpoint-reconstructible from config; do not rely on an implicit source edit with no corresponding config field.
- Do not describe any gains as “pooling alone worked”; the experiment does not isolate pooling from side-basis capacity.
- Keep `side_z_dim=32` for `v28`; do not continue the `side_z_dim=8` line here.
- Keep the fallback laterality-contrast idea out of `v28`; it is a separate next-step branch only if this plan fails.

## Task 1: Add Fixed 4-Block Side-Aware Pooling To The Early Side Path

**Files:**
- Modify: `scripts/lq/model/distnet.py`
- Modify: `scripts/lq/analyze_checkpoint.py`
- Test: ad-hoc shape/contract smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing side-pooling contract smoke**

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
    side_pooling="fixed_block4_diag",
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
    private_pool_size=1,
    side_z_dim=32,
)

side_feats = torch.randn(5, 32, 15, 15)
pooled = model._pool_side_tokens_fixed_blocks(side_feats)

assert pooled.shape == (5, 128)
print("side-aware-pool-shape-ok")
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
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
    side_pooling="fixed_block4_diag",
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
    private_pool_size=1,
    side_z_dim=32,
)

side_feats = torch.randn(5, 32, 15, 15)
pooled = model._pool_side_tokens_fixed_blocks(side_feats)

assert pooled.shape == (5, 128)
print("side-aware-pool-shape-ok")
PY
```

Expected: attribute error or shape failure because the fixed-block side pooling helper does not exist yet.

- [ ] **Step 3: Implement the minimal fixed-block pooling helper and wire it into the early side path**

In `scripts/lq/model/distnet.py`:

1. Add a helper dedicated to the side-aware readout, for example:

```python
def _pool_side_tokens_fixed_blocks(self, side_feats: torch.Tensor) -> torch.Tensor:
    blocks = (
        (slice(0, 3), slice(0, 3)),
        (slice(3, 6), slice(3, 6)),
        (slice(6, 10), slice(6, 10)),
        (slice(10, 15), slice(10, 15)),
    )
    tokens = []
    for row_slice, col_slice in blocks:
        block = side_feats[:, :, row_slice, col_slice]
        tokens.append(block.mean(dim=(2, 3)))
    return torch.cat(tokens, dim=1)
```

2. In `_forward_early_branch(...)`, branch on the explicit pooling mode:

```python
if self.side_pooling == "fixed_block4_diag":
    side_pooled = self._pool_side_tokens_fixed_blocks(side_feats)
else:
    side_pooled = self.side_pool(side_feats).flatten(1)
```

This preserves backward compatibility for older checkpoints and makes `v28` an explicit config-selected architecture variant.

3. Keep:

- `free_pool` behavior unchanged
- `private_pool` behavior unchanged
- existing side head / side basis / side path outputs unchanged except for input dim

4. Make sure the early-branch constructor builds `self.side_head` against the new side pooled dim when `side_pooling == "fixed_block4_diag"`:

```python
self.side_pooled_dim = hidden_dim * 4
```

5. In `scripts/lq/analyze_checkpoint.py`, read `side_pooling` from checkpoint config and pass it into `DistNet(...)`:

```python
side_pooling = str(config.get("side_pooling", "masked_mean"))
...
model = DistNet(
    ...
    side_pooling=side_pooling,
    ...
)
```

Do not change the shared trunk or quantizer code.

- [ ] **Step 4: Run the side-pooling shape smoke again**

Run the same command from Step 2.

Expected: prints `side-aware-pool-shape-ok`.

- [ ] **Step 5: Run an end-to-end forward smoke on grouped input**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
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
    side_pooling="fixed_block4_diag",
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
    private_pool_size=1,
    side_z_dim=32,
)

x = torch.randn(3, 4, 1, 119, 119)
out = model(x, return_group_pooled=True)

assert out["side_path_usage"].shape == (3, 4, 4)
assert out["side_path_representation"].shape == (3, 4, 4)
assert out["group_pooled_side_rep"].shape == (3, 4)
print("side-aware-forward-ok")
PY
```

Expected: prints `side-aware-forward-ok`.

- [ ] **Step 6: Commit**

```bash
git add scripts/lq/model/distnet.py scripts/lq/analyze_checkpoint.py
git commit -m "lq: add side-aware pooling to early side path"
```

## Task 2: Add The Canonical `v28` Training Preset

**Files:**
- Create: `scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh`
- Test: existence + diff + shell-parse checks

- [ ] **Step 1: Verify the new preset does not exist yet**

Run:

```bash
test -f scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh
```

Expected: non-zero exit because the file does not exist yet.

- [ ] **Step 2: Create the preset from `v26` with only the intended deltas**

Create `scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh` based on `scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh` and keep:

- `data_roots=data/win20-step20/IMR,data/win20-step20/TT`
- `epochs=50`
- `batch_size=64`
- `early_branch_factorization=True`
- `quantizer_type=residual_fsq`
- `basis_orthogonalization=global_qr`
- the same conda activation pattern with `set +u` around `conda activate dl`

Change only the intended candidate settings:

```bash
  --side_pooling=fixed_block4_diag \
  --side_basis_count=4 \
  --side_z_dim=32 \
  --output_dir=outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50
```

Keep `side_loss_weight=0.3` and the rest of the baseline hyperparameters unchanged.

- [ ] **Step 3: Run a shell parse check**

Run:

```bash
bash -n scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh
```

Expected: exits `0`.

- [ ] **Step 4: Diff the new preset against `v26` to confirm only expected user-facing deltas**

Run:

```bash
diff -u scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh
```

Expected: the meaningful differences are `side_pooling=fixed_block4_diag`, `side_basis_count=4`, explicit `side_z_dim=32` if needed, and the `output_dir`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh
git commit -m "lq: add v28 side-aware pooling preset"
```

## Task 3: Smoke-Test `v28` With An Explicit Short Run

**Files:**
- Read: `scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh`
- Produce: `outputs/lq_x_mouth_v28_side_aware_pooling_probe_smoke/`
- Test: 3-epoch smoke run + analysis

- [ ] **Step 1: Run an explicit 3-epoch smoke command that matches the `v28` preset**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --group_size=4 \
  --mode=x \
  --region=mouth \
  --basis_size=119 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x.npy \
  --hidden_dim=32 \
  --private_dim=32 \
  --recon_weight=1.0 \
  --shared_recon_weight=1.0 \
  --lq_weight=10.0 \
  --orth_weight=0.1 \
  --basis_l1_weight=1.0 \
  --residual_weight=0.02 \
  --private_residual_weight=0.05 \
  --private_residual_max_l1=0.5 \
  --shared_basis_soft_mixing=True \
  --shared_basis_anchor_bias=2.0 \
  --shared_basis_topk=2 \
  --quantizer_type=residual_fsq \
  --fsq_preserve_symmetry=True \
  --basis_orthogonalization=global_qr \
  --epochs=3 \
  --batch_size=64 \
  --side_semantic_enabled=True \
  --side_pooling=fixed_block4_diag \
  --side_basis_count=4 \
  --side_z_dim=32 \
  --side_loss_weight=0.3 \
  --use_dataset_aux=False \
  --early_branch_factorization=True \
  --free_pool_size=2 \
  --side_pool_size=2 \
  --private_pool_size=1 \
  --output_dir=outputs/lq_x_mouth_v28_side_aware_pooling_probe_smoke
```

Expected: the smoke run uses the same effective config as the canonical `v28` preset except `epochs=3` and `output_dir`, and writes `outputs/lq_x_mouth_v28_side_aware_pooling_probe_smoke/best.pt`.

- [ ] **Step 2: Analyze the smoke checkpoint**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v28_side_aware_pooling_probe_smoke/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --output_dir=outputs/lq_x_mouth_v28_side_aware_pooling_probe_smoke/analysis
```

Expected: completes and writes `summary.json`.

- [ ] **Step 3: Sanity-check smoke metrics before the full run**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path("outputs/lq_x_mouth_v28_side_aware_pooling_probe_smoke/analysis/summary.json").read_text()
)
print("val_recon", summary["val_metrics"]["recon"])
print("val_side_group", summary["val_metrics"]["side_group"])
print("side_from_side_rep_acc", summary["side_probe"]["side_from_side_rep_acc"])
print("dataset_from_side_rep_acc", summary["dataset_probe"]["dataset_from_side_rep_acc"])
PY
```

Expected: all metrics are finite, no shape failures, and the side probe pipeline remains analyzable with `side_pooling=fixed_block4_diag` and `side_basis_count=4`.

## Task 4: Run The Full `v28` Probe And Compare Against Pinned `v26`

**Files:**
- Run: `scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh`
- Produce: `outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50/analysis/summary.json`
- Test: one reproducible comparison snippet

- [ ] **Step 1: Launch the canonical 50-epoch run**

Run:

```bash
bash scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh
```

Expected: completes and writes `outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50/best.pt`.

- [ ] **Step 2: Analyze the best checkpoint under the existing canonical side-rep rule**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --output_dir=outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50/analysis
```

Expected: writes `summary.json`.

- [ ] **Step 3: Compare `v28` against the pinned baseline in one script**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
import json
from pathlib import Path

v26 = json.loads(
    Path("outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/summary.json").read_text()
)
v28 = json.loads(
    Path("outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50/analysis/summary.json").read_text()
)

print("v26 side_from_side_rep_acc", v26["side_probe"]["side_from_side_rep_acc"])
print("v28 side_from_side_rep_acc", v28["side_probe"]["side_from_side_rep_acc"])
print("v26 side_from_free_rep_acc", v26["side_probe"]["side_from_free_rep_acc"])
print("v28 side_from_free_rep_acc", v28["side_probe"]["side_from_free_rep_acc"])
print("v26 dataset_from_side_rep_acc", v26["dataset_probe"]["dataset_from_side_rep_acc"])
print("v28 dataset_from_side_rep_acc", v28["dataset_probe"]["dataset_from_side_rep_acc"])
print("v26 val_recon", v26["val_metrics"]["recon"])
print("v28 val_recon", v28["val_metrics"]["recon"])
print("v26 val_side_group", v26["val_metrics"]["side_group"])
print("v28 val_side_group", v28["val_metrics"]["side_group"])
PY
```

Expected success criteria:

- `v28 side_from_side_rep_acc > v28 side_from_free_rep_acc`
- `v28 side_from_side_rep_acc > 0.8227` is preferred
- `v28 dataset_from_side_rep_acc < 0.8182` is preferred
- `v28 val_recon - 0.2873 <= 0.01`
- `v28 val_side_group - 0.5365 <= 0.10`

Interpretation rule if the preferred criteria fail:

- Conclude that `side-aware pooling + side_basis_count=4` did not adequately rescue the side path under the fixed trunk.
- That failure is the trigger to move to the fallback laterality-contrast-token variant.

- [ ] **Step 4: Commit the executable changes**

```bash
git add scripts/lq/model/distnet.py scripts/lq/analyze_checkpoint.py scripts/lq/run_train_x_mouth_v28_side_aware_pooling_probe.sh
git commit -m "lq: add side-aware pooling probe"
```

## Task 5: Update Research Notes

**Files:**
- Modify: `RESEARCH_PROGRESS.md`
- Test: manual diff review

- [ ] **Step 1: Add a short progress note for `v28`**

Include:

- the exact candidate path:
  - `outputs/lq_x_mouth_v28_side_aware_pooling_probe_win20_e50`
- the key comparison values versus the pinned `v26 reanalyzed` baseline
- whether the joint `side-aware pooling + side_basis_count=4` probe helped
- whether to escalate to the fallback laterality-contrast-token version

- [ ] **Step 2: Review the diff**

Run:

```bash
git diff -- RESEARCH_PROGRESS.md
```

Expected: a concise note with concrete results, not a restatement of the full spec.

- [ ] **Step 3: Commit**

```bash
git add RESEARCH_PROGRESS.md
git commit -m "docs: record v28 side-aware pooling results"
```
