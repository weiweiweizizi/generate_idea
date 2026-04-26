# LQ Side Basis Rep Tightening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the `side` branch baseline by forcing analysis-time canonical side representations to be recomputed from `side_path_representation`, then train and compare a new `v27` run with `side_z_dim=8` against a reanalyzed `v26` reference checkpoint.

**Architecture:** Keep the `v26 early-branch` training graph intact. The only code-path change is in `scripts/lq/analyze_checkpoint.py`: canonical side probes must use `masked_mean_per_sequence(side_path_representation, valid_mask)` as the source of truth instead of any cached `group_pooled_side_rep`. The new `v27` line is introduced as a shell preset that reuses the existing train/config/distnet surface and changes only `side_z_dim` plus output naming.

**Tech Stack:** Python, PyTorch, `vector-quantize-pytorch`, Fire CLI, Bash, existing `scripts/lq` train/analyze workflow, `dl` conda environment

---

## File Map

- Modify: `scripts/lq/analyze_checkpoint.py`
  - Add an explicit analysis-side canonical pooling path that recomputes side reps from `side_path_representation` and `valid_mask`, while preserving fallback behavior for non-canonical tensors and old checkpoints.
- Create: `scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh`
  - Clone the `v26` early-branch preset and set `--side_z_dim=8`, plus a distinct `output_dir`.
- Modify: `RESEARCH_PROGRESS.md`
  - Record the `v26` reanalysis result, the `v27` 50-epoch result, and the next decision point.

## Pinned References

- Approved spec: `docs/superpowers/specs/2026-04-20-lq-side-basis-rep-tightening-design.md`
- Pinned baseline checkpoint: `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/best.pt`
- Existing baseline training entrypoint: `scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh`
- New training entrypoint to create in this plan: `scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh`
- Dataset for all comparisons: `data/win20-step20/IMR,data/win20-step20/TT`

## Constraints

- Every Python command must run under `dl`:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python ...
```

- Do not change the `v26` training script, model graph, or loss surface.
- Do not touch `scripts/lq/model/distnet.py`, `scripts/lq/train.py`, or `scripts/lq/training/config.py` unless implementation proves a real gap; the spec delta is intentionally analysis + preset only.
- Canonical `side` representation in analysis must be recomputed from `side_path_representation`; do not rely on cached `group_pooled_side_rep`.
- `free` canonical representation stays on the current logic in this round.
- Comparison must use:
  - `v26` reanalyzed under the new canonical side-rep rule
  - `v27` analyzed under the same rule

## Task 1: Force Canonical Side Rep Recompute In Analysis

**Files:**
- Modify: `scripts/lq/analyze_checkpoint.py`
- Test: ad-hoc helper smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing helper-contract smoke**

```python
import torch

from scripts.lq.analyze_checkpoint import (
    masked_mean_per_sequence,
    pool_group_tensor,
)

valid_mask = torch.tensor(
    [
        [True, True, False],
        [True, False, False],
    ]
)
sequence = torch.tensor(
    [
        [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
        [[2.0, 4.0], [200.0, 200.0], [300.0, 300.0]],
    ]
)
cached = torch.full((2, 2), 99.0)

result = pool_group_tensor(
    {
        "group_pooled_side_rep": cached,
        "side_path_representation": sequence,
    },
    pooled_key="group_pooled_side_rep",
    sequence_key="side_path_representation",
    valid_mask=valid_mask,
    require_pooled=False,
    force_sequence=True,
)
expected = masked_mean_per_sequence(sequence, valid_mask)

assert torch.allclose(result, expected)
assert not torch.allclose(result, cached)
print("canonical-side-repool-ok")
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
import torch

from scripts.lq.analyze_checkpoint import (
    masked_mean_per_sequence,
    pool_group_tensor,
)

valid_mask = torch.tensor(
    [
        [True, True, False],
        [True, False, False],
    ]
)
sequence = torch.tensor(
    [
        [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
        [[2.0, 4.0], [200.0, 200.0], [300.0, 300.0]],
    ]
)
cached = torch.full((2, 2), 99.0)

result = pool_group_tensor(
    {
        "group_pooled_side_rep": cached,
        "side_path_representation": sequence,
    },
    pooled_key="group_pooled_side_rep",
    sequence_key="side_path_representation",
    valid_mask=valid_mask,
    require_pooled=False,
    force_sequence=True,
)
expected = masked_mean_per_sequence(sequence, valid_mask)

assert torch.allclose(result, expected)
assert not torch.allclose(result, cached)
print("canonical-side-repool-ok")
PY
```

Expected: failure because `pool_group_tensor(...)` does not accept `force_sequence` yet, or it still returns the cached pooled tensor.

- [ ] **Step 3: Implement the minimal analysis-side source-of-truth change**

In `scripts/lq/analyze_checkpoint.py`:

```python
def pool_group_tensor(
    outputs: dict,
    *,
    pooled_key: str,
    sequence_key: str,
    valid_mask: torch.Tensor,
    require_pooled: bool,
    empty_feature_dim: int | None = None,
    force_sequence: bool = False,
) -> torch.Tensor:
    sequence_tensor = outputs.get(sequence_key)

    if force_sequence:
        if sequence_tensor is None:
            raise RuntimeError(f"{sequence_key} is required to recompute canonical rep")
        return masked_mean_per_sequence(sequence_tensor, valid_mask)

    pooled = outputs.get(pooled_key)
    if pooled is not None:
        return pooled
    ...
```

Use it in `collect_group_representations(...)` only for canonical side reps:

```python
group_side_rep = pool_group_tensor(
    outputs,
    pooled_key="group_pooled_side_rep",
    sequence_key="side_path_representation",
    valid_mask=valid_mask,
    require_pooled=False,
    force_sequence=True,
)
```

Keep `group_free_rep`, `group_private_rep`, and latent diagnostics on the current fallback logic.

- [ ] **Step 4: Run the helper smoke again**

Run the same command from Step 2.

Expected: prints `canonical-side-repool-ok`.

- [ ] **Step 5: Run pinned `v26` analysis once to verify the new path works on a real checkpoint**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --output_dir=outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep
```

Expected: completes without requiring retraining and writes:

- `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/summary.json`
- `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/group_level_representations.npz`

- [ ] **Step 6: Commit**

```bash
git add scripts/lq/analyze_checkpoint.py
git commit -m "lq: recompute canonical side reps in analysis"
```

## Task 2: Add The `v27` Training Entrypoint

**Files:**
- Create: `scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh`
- Test: existence + shell-parse check

- [ ] **Step 1: Verify the new preset file does not exist yet**

Run:

```bash
test -f scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh
```

Expected: non-zero exit because the file does not exist yet.

- [ ] **Step 2: Create the new preset by cloning `v26` and tightening `side_z_dim`**

Create `scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh` with the same command line as `scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh`, plus:

```bash
  --side_z_dim=8 \
  --output_dir=outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50
```

Keep:

- `set +u` around `conda activate dl`
- dataset roots on `data/win20-step20/IMR,data/win20-step20/TT`
- `batch_size=64`
- `epochs=50`
- the existing `early_branch_factorization=True` and residual-FSQ settings

- [ ] **Step 3: Run a shell-parse smoke**

Run:

```bash
bash -n scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh
```

Expected: exits `0` with no output.

- [ ] **Step 4: Commit**

```bash
git add scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh
git commit -m "lq: add v27 side basis tightening preset"
```

## Task 3: Reanalyze `v26` Under The New Canonical Rule

**Files:**
- Read/produce: `outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/summary.json`
- Test: ad-hoc summary inspection via `python - <<'PY'`

- [ ] **Step 1: Inspect the reanalysis summary and extract the comparable probe numbers**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
import json
from pathlib import Path

summary_path = Path("outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/summary.json")
summary = json.loads(summary_path.read_text())

print("side_from_side_rep_acc", summary["side_probe"]["side_from_side_rep_acc"])
print("side_from_free_rep_acc", summary["side_probe"]["side_from_free_rep_acc"])
print("dataset_from_side_rep_acc", summary["dataset_probe"]["dataset_from_side_rep_acc"])
print("dataset_from_free_rep_acc", summary["dataset_probe"]["dataset_from_free_rep_acc"])
print("dataset_from_private_rep_acc", summary["dataset_probe"]["dataset_from_private_rep_acc"])
print("val_recon", summary["val_metrics"]["recon"])
print("val_side_group", summary["val_metrics"]["side_group"])
PY
```

Expected: prints seven floats and confirms the reanalysis artifact is readable.

- [ ] **Step 2: Save a short comparison note locally for the next task**

Record these numbers in your task notes before training `v27`:

- `side_from_side_rep_acc`
- `side_from_free_rep_acc`
- `dataset_from_side_rep_acc`
- `val_metrics["recon"]`
- `val_metrics["side_group"]`

This is the baseline comparison target for the full `v27` run.

## Task 4: Smoke-Test `v27` Before The Full 50-Epoch Run

**Files:**
- Read: `scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh`
- Produce: `outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_smoke/`
- Test: short 3-epoch training command + analysis

- [ ] **Step 1: Generate a temporary 3-epoch smoke wrapper from the real `v27` preset and run it**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
from pathlib import Path

src = Path("scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh").read_text()
src = src.replace("--epochs=50 \\", "--epochs=3 \\")
src = src.replace(
    "outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50",
    "outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_smoke",
)
Path("scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe_smoke.sh").write_text(src)
print("scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe_smoke.sh")
PY

bash scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe_smoke.sh
```

Expected: the smoke run exercises the real `v27` preset contents with only `epochs` and `output_dir` overridden, writes `outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_smoke/best.pt`, and does not show shape/config mismatches caused by the smaller `side_z_dim`.

- [ ] **Step 2: Run smoke analysis**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_smoke/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --output_dir=outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_smoke/analysis
```

Expected: completes and writes a `summary.json`.

- [ ] **Step 3: Sanity-check smoke metrics before launching 50 epochs**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path("outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_smoke/analysis/summary.json").read_text()
)
print("val_recon", summary["val_metrics"]["recon"])
print("val_side_group", summary["val_metrics"]["side_group"])
print("side_from_side_rep_acc", summary["side_probe"]["side_from_side_rep_acc"])
print("dataset_from_side_rep_acc", summary["dataset_probe"]["dataset_from_side_rep_acc"])
PY
```

Expected: metrics are finite and broadly in-family with `v26`; no NaNs and no catastrophic collapse.

## Task 5: Run The Full `v27` Baseline And Compare It To Reanalyzed `v26`

**Files:**
- Run: `scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh`
- Produce: `outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50/`
- Produce: `outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50/analysis/summary.json`
- Test: ad-hoc comparison snippet via `python - <<'PY'`

- [ ] **Step 1: Launch the 50-epoch run from the dedicated preset**

Run:

```bash
bash scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh
```

Expected: completes and writes `outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50/best.pt`.

- [ ] **Step 2: Analyze the `v27` best checkpoint under the same canonical rule**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --output_dir=outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50/analysis
```

Expected: completes and writes `summary.json`.

- [ ] **Step 3: Compare `v27` against reanalyzed `v26` with one reproducible script**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
import json
from pathlib import Path

v26 = json.loads(
    Path("outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/analysis_v27_side_rep/summary.json").read_text()
)
v27 = json.loads(
    Path("outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50/analysis/summary.json").read_text()
)

print("v26 side_from_side_rep_acc", v26["side_probe"]["side_from_side_rep_acc"])
print("v27 side_from_side_rep_acc", v27["side_probe"]["side_from_side_rep_acc"])
print("v26 side_from_free_rep_acc", v26["side_probe"]["side_from_free_rep_acc"])
print("v27 side_from_free_rep_acc", v27["side_probe"]["side_from_free_rep_acc"])
print("v26 dataset_from_side_rep_acc", v26["dataset_probe"]["dataset_from_side_rep_acc"])
print("v27 dataset_from_side_rep_acc", v27["dataset_probe"]["dataset_from_side_rep_acc"])
print("v26 val_recon", v26["val_metrics"]["recon"])
print("v27 val_recon", v27["val_metrics"]["recon"])
print("v26 val_side_group", v26["val_metrics"]["side_group"])
print("v27 val_side_group", v27["val_metrics"]["side_group"])
PY
```

Expected:

- `v27 side_from_side_rep_acc > v27 side_from_free_rep_acc`
- `v27 dataset_from_side_rep_acc < v26 dataset_from_side_rep_acc` is preferred
- `v27 val_recon - v26 val_recon <= 0.01` is preferred
- `v27 val_side_group - v26 val_side_group <= 0.10` is preferred

- [ ] **Step 4: Commit the executable changes after the run is validated**

```bash
git add scripts/lq/analyze_checkpoint.py scripts/lq/run_train_x_mouth_v27_side_basis_rep_tight_probe.sh
git commit -m "lq: add side basis tightening baseline"
```

## Task 6: Update Research Notes

**Files:**
- Modify: `RESEARCH_PROGRESS.md`
- Test: manual diff review

- [ ] **Step 1: Add a short progress note for this round**

Include:

- the pinned `v26` reanalysis path and headline metrics
- the `v27` run path and headline metrics
- whether `dataset_from_side_rep_acc` moved in the desired direction
- whether reconstruction / side-group costs stayed within the tolerated drift
- the next branch decision:
  - keep tightening side representation
  - or move to the next intervention because the leak is basis-level rather than latent-capacity-level

- [ ] **Step 2: Review the diff for signal only**

Run:

```bash
git diff -- RESEARCH_PROGRESS.md
```

Expected: the note is brief, concrete, and does not duplicate the full spec or plan.

- [ ] **Step 3: Commit**

```bash
git add RESEARCH_PROGRESS.md
git commit -m "docs: record v27 side basis tightening results"
```
