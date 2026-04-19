# scripts/lq Refactor Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the active `scripts/lq` training, dataset, and model core into smaller modules while preserving the current experiment workflow, CLI entrypoints, and output semantics.

**Architecture:** Keep `scripts/lq/train.py` and `scripts/lq/analyze_checkpoint.py` as stable entrypoints, then progressively extract focused modules under `scripts/lq/training/`, `scripts/lq/data/`, and `scripts/lq/model/`. The first pass preserves behavior first and only performs local cleanup when it improves clarity without changing current training semantics.

**Tech Stack:** Python, PyTorch, `fire`, `tqdm`, `pandas`, `numpy`, `vector-quantize-pytorch`

---

## File Structure Lock-In

### Existing files to keep as public entrypoints

- `scripts/lq/train.py`
- `scripts/lq/analyze_checkpoint.py`
- `scripts/lq/model/network.py`

### New files to create in round 1

- `scripts/lq/training/__init__.py`
- `scripts/lq/training/config.py`
- `scripts/lq/training/data.py`
- `scripts/lq/training/losses.py`
- `scripts/lq/training/engine.py`
- `scripts/lq/training/checkpoint.py`
- `scripts/lq/data/__init__.py`
- `scripts/lq/data/specs.py`
- `scripts/lq/data/io.py`
- `scripts/lq/data/samples.py`
- `scripts/lq/data/datasets.py`
- `scripts/lq/model/encoder.py`
- `scripts/lq/model/basis.py`
- `scripts/lq/model/quantizers.py`
- `scripts/lq/model/heads.py`
- `scripts/lq/model/distnet.py`

### Existing files to modify in round 1

- `scripts/lq/train.py`
- `scripts/lq/datasets.py`
- `scripts/lq/analyze_checkpoint.py`
- `scripts/lq/model/network.py`

### Compatibility rules that every task must preserve

- `python scripts/lq/train.py ...` must continue to work.
- `python scripts/lq/analyze_checkpoint.py ...` must continue to work.
- Existing batch keys from `FacialMotionSequenceDataset` must remain stable.
- Existing `DistNet` constructor arguments and output keys must remain stable.
- Existing checkpoint fields `config`, `train_metrics`, and `val_metrics` must remain stable.
- Existing shell scripts under `scripts/lq/run_train_x_mouth_v*.sh` must not require bulk edits.

---

### Task 1: Create the `training/` package skeleton

**Files:**
- Create: `scripts/lq/training/__init__.py`
- Create: `scripts/lq/training/config.py`
- Create: `scripts/lq/training/data.py`
- Create: `scripts/lq/training/losses.py`
- Create: `scripts/lq/training/engine.py`
- Create: `scripts/lq/training/checkpoint.py`
- Modify: `scripts/lq/train.py`

- [ ] **Step 1: Create the package marker**

Create `scripts/lq/training/__init__.py` with a short module docstring only.

- [ ] **Step 2: Extract config validation helpers**

Move pure validation and normalization logic out of `scripts/lq/train.py` into
`scripts/lq/training/config.py`, including:

- side weight default resolution
- `action_basis_init_path` existence checks
- region-to-basis-size checks
- config dict preparation helper

Keep argument names and defaults unchanged at the entrypoint.

- [ ] **Step 3: Extract dataset-building helpers**

Move these functions into `scripts/lq/training/data.py`:

- `build_specs`
- `build_datasets`
- dataloader construction helper for train/val loaders

Keep data-root parsing behavior unchanged.

- [ ] **Step 4: Extract masked reduction and batch-step logic**

Move these functions into `scripts/lq/training/losses.py`:

- `masked_mean`
- `step_model`

Preserve metric keys exactly:

- `loss`
- `recon`
- `shared_recon`
- `lq`
- `orth`
- `basis_l1`
- `residual`
- `scaled_residual`
- optional side and dataset metrics

- [ ] **Step 5: Extract epoch and memory-check helpers**

Move these functions into `scripts/lq/training/engine.py`:

- `tensor_memory_mib`
- `run_batch_memory_validation`
- `run_epoch`

Keep the printed memory-check output format stable enough for current logs.

- [ ] **Step 6: Extract checkpoint saving helper**

Create `scripts/lq/training/checkpoint.py` with a single helper responsible for
saving the `best.pt` payload using the same field names as today.

- [ ] **Step 7: Reduce `scripts/lq/train.py` to a thin entrypoint**

Refactor `scripts/lq/train.py` so it:

- keeps the same `train(...)` signature
- imports helper functions from `scripts/lq/training/`
- assembles config, model, data, and optimizer
- delegates the step logic and epoch loop

- [ ] **Step 8: Run a syntax smoke check**

Run:

```bash
python -m py_compile scripts/lq/train.py scripts/lq/training/config.py scripts/lq/training/data.py scripts/lq/training/losses.py scripts/lq/training/engine.py scripts/lq/training/checkpoint.py
```

Expected: no output, exit code `0`.

- [ ] **Step 9: Run a train-entry smoke**

Run one short command using the existing CLI shape:

```bash
python scripts/lq/train.py --data_roots=data/win20-step20/IMR,data/win20-step20/TT --epochs=1 --batch_size=64 --group_size=4 --mode=x --region=mouth --basis_size=119 --action_basis_init_path=scripts/lq/init_basis/basis_x.npy --hidden_dim=32 --private_dim=32 --pool_size=1 --recon_weight=1.0 --shared_recon_weight=1.0 --lq_weight=10.0 --orth_weight=0.1 --basis_l1_weight=1.0 --residual_weight=0.02 --side_weight=0.0 --side_cont_weight=0.0 --side_disc_weight=0.0 --private_residual_weight=0.05 --private_residual_max_l1=0.5 --shared_basis_soft_mixing=True --shared_basis_anchor_bias=2.0 --shared_basis_topk=2 --quantizer_type=residual_fsq --fsq_preserve_symmetry=True --basis_orthogonalization=global_qr --use_dataset_aux=False --output_dir=outputs/lq_refactor_round1_train_smoke
```

Expected:

- memory-check prints successfully
- one epoch completes
- `outputs/lq_refactor_round1_train_smoke/best.pt` is created

- [ ] **Step 10: Commit**

```bash
git add scripts/lq/train.py scripts/lq/training
git commit -m "refactor: split lq training entrypoint helpers"
```

---

### Task 2: Create the `data/` package and preserve dataset semantics

**Files:**
- Create: `scripts/lq/data/__init__.py`
- Create: `scripts/lq/data/specs.py`
- Create: `scripts/lq/data/io.py`
- Create: `scripts/lq/data/samples.py`
- Create: `scripts/lq/data/datasets.py`
- Modify: `scripts/lq/datasets.py`
- Modify: `scripts/lq/train.py`
- Modify: `scripts/lq/analyze_checkpoint.py`

- [ ] **Step 1: Move dataset spec and subject helpers**

Create `scripts/lq/data/specs.py` and move:

- `DatasetSpec`
- `create_side_label`
- `create_severity_label`
- `subject_split`

Keep behavior identical.

- [ ] **Step 2: Move metadata and matrix IO helpers**

Create `scripts/lq/data/io.py` and move:

- `_infer_subject_width`
- `_load_metadata`
- `_get_deleted_column`
- `_zero_pad_array`
- global signed scale estimation logic currently inside
  `FacialMotionSequenceDataset.compute_global_scale`

Make these helpers importable without needing dataset-class construction.

- [ ] **Step 3: Move common sample-dict construction**

Create `scripts/lq/data/samples.py` and move the sample-building logic out of
`_BaseFacialMotionDataset._build_sample_dict`.

Keep all existing output keys unchanged.

- [ ] **Step 4: Move dataset classes into the package**

Create `scripts/lq/data/datasets.py` and move:

- `_BaseFacialMotionDataset`
- `FacialMotionDataset`
- `FacialMotionSequenceDataset`

Keep:

- grouping rule
- `valid_mask`
- `padding_mask`
- `sample_source`
- `sample_ids`
- `window_indices`
- `prev_window_indices`

exactly compatible.

- [ ] **Step 5: Turn `scripts/lq/datasets.py` into a compatibility shim**

Replace the current file body with imports that re-export:

- `DatasetSpec`
- `FacialMotionDataset`
- `FacialMotionSequenceDataset`
- `subject_split`
- label helpers if still referenced externally

- [ ] **Step 6: Update training and analysis imports**

Modify:

- `scripts/lq/train.py`
- `scripts/lq/analyze_checkpoint.py`

so they import through the new package layout cleanly.

- [ ] **Step 7: Run syntax smoke check**

Run:

```bash
python -m py_compile scripts/lq/datasets.py scripts/lq/analyze_checkpoint.py scripts/lq/data/specs.py scripts/lq/data/io.py scripts/lq/data/samples.py scripts/lq/data/datasets.py
```

Expected: no output, exit code `0`.

- [ ] **Step 8: Run dataset-shape smoke**

Run:

```bash
python - <<'PY'
from scripts.lq.data.specs import DatasetSpec, subject_split
from scripts.lq.data.datasets import FacialMotionSequenceDataset
from pathlib import Path

spec = DatasetSpec(root=Path("data/win20-step20/IMR"), dataset_label=0, dataset_name="IMR")
train_subjects, _ = subject_split(spec, val_ratio=0.2, seed=42)
dataset = FacialMotionSequenceDataset(spec, train_subjects[:8], mode="x", region="mouth", group_size=4)
sample = dataset[0]
print(sample["images"].shape)
print(sample["valid_mask"].shape)
print(sample["padding_mask"].shape)
print(sorted(k for k in sample.keys()))
PY
```

Expected:

- image tensor shape is `(4, 1, 119, 119)`
- masks have shape `(4,)`
- sample keys match the current dataset contract

- [ ] **Step 9: Re-run the train smoke**

Repeat the one-epoch train smoke from Task 1 and verify it still succeeds after
the dataset extraction.

- [ ] **Step 10: Commit**

```bash
git add scripts/lq/datasets.py scripts/lq/analyze_checkpoint.py scripts/lq/train.py scripts/lq/data
git commit -m "refactor: split lq dataset metadata and grouping logic"
```

---

### Task 3: Split `DistNet` internals into focused model modules

**Files:**
- Create: `scripts/lq/model/encoder.py`
- Create: `scripts/lq/model/basis.py`
- Create: `scripts/lq/model/quantizers.py`
- Create: `scripts/lq/model/heads.py`
- Create: `scripts/lq/model/distnet.py`
- Modify: `scripts/lq/model/network.py`
- Modify: `scripts/lq/analyze_checkpoint.py`
- Modify: `scripts/lq/train.py`

- [ ] **Step 1: Extract encoder construction**

Move the CNN stem and residual encoder creation into
`scripts/lq/model/encoder.py`.

This should include:

- initial conv
- `layer1`
- `layer2`
- `layer3`
- adaptive pooling helper or module assembly

- [ ] **Step 2: Extract basis operations**

Move these responsibilities into `scripts/lq/model/basis.py`:

- basis init loading
- symmetric zero-diagonal constraint
- normalize / `level_qr` / `global_qr`
- row-wise QR helper
- orthogonality loss
- basis L1 loss
- basis splitting helper

- [ ] **Step 3: Extract quantizer wrapper**

Create `scripts/lq/model/quantizers.py` to unify:

- `LatentQuantize`
- `FSQ`
- residual-FSQ stack behavior
- shared-latent quantization API normalization
- index decoding helper

The wrapper must preserve the current outputs expected by `DistNet.forward()`.

- [ ] **Step 4: Extract heads**

Create `scripts/lq/model/heads.py` for:

- shared head
- private head
- shared coefficient heads
- shared basis heads
- side classifiers
- dataset auxiliary classifiers

- [ ] **Step 5: Move `DistNet` into `distnet.py`**

Create `scripts/lq/model/distnet.py` and rebuild `DistNet` there using the
new submodules.

Rules:

- constructor signature must stay compatible
- output dict keys must stay unchanged
- sequence flatten/reshape behavior must stay unchanged
- private residual limiting behavior must stay unchanged

- [ ] **Step 6: Turn `scripts/lq/model/network.py` into a compatibility layer**

Reduce `network.py` so it re-exports `DistNet` from `distnet.py`. Keep any
minimal compatibility helpers only if still required externally.

- [ ] **Step 7: Run syntax smoke check**

Run:

```bash
python -m py_compile scripts/lq/model/network.py scripts/lq/model/distnet.py scripts/lq/model/encoder.py scripts/lq/model/basis.py scripts/lq/model/quantizers.py scripts/lq/model/heads.py
```

Expected: no output, exit code `0`.

- [ ] **Step 8: Run forward-contract smoke**

Run:

```bash
python - <<'PY'
import torch
from scripts.lq.model.network import DistNet

model = DistNet(
    levels=(2, 3, 6),
    basis_size=119,
    hidden_dim=32,
    private_dim=32,
    pool_size=1,
    action_basis_init_path="scripts/lq/init_basis/basis_x.npy",
    quantizer_type="residual_fsq",
    basis_orthogonalization="global_qr",
    shared_basis_soft_mixing=True,
    shared_basis_anchor_bias=2.0,
    shared_basis_topk=2,
)
x = torch.randn(2, 4, 1, 119, 119)
out = model(x)
print(sorted(out.keys()))
print(out["reconstructed"].shape)
print(out["action_reconstruction"].shape)
print(out["private_residual"].shape)
PY
```

Expected:

- output keys match the current contract
- tensor shapes restore to `B x T x 1 x H x W`

- [ ] **Step 9: Re-run train and analysis smoke**

Run:

```bash
python scripts/lq/train.py --data_roots=data/win20-step20/IMR,data/win20-step20/TT --epochs=1 --batch_size=64 --group_size=4 --mode=x --region=mouth --basis_size=119 --action_basis_init_path=scripts/lq/init_basis/basis_x.npy --hidden_dim=32 --private_dim=32 --pool_size=1 --recon_weight=1.0 --shared_recon_weight=1.0 --lq_weight=10.0 --orth_weight=0.1 --basis_l1_weight=1.0 --residual_weight=0.02 --side_weight=0.0 --side_cont_weight=0.0 --side_disc_weight=0.0 --private_residual_weight=0.05 --private_residual_max_l1=0.5 --shared_basis_soft_mixing=True --shared_basis_anchor_bias=2.0 --shared_basis_topk=2 --quantizer_type=residual_fsq --fsq_preserve_symmetry=True --basis_orthogonalization=global_qr --use_dataset_aux=False --output_dir=outputs/lq_refactor_round1_model_smoke
python scripts/lq/analyze_checkpoint.py --checkpoint_path=outputs/lq_refactor_round1_model_smoke/best.pt --data_roots=data/win20-step20/IMR,data/win20-step20/TT
```

Expected:

- train smoke succeeds
- analysis writes `analysis/summary.json`

- [ ] **Step 10: Commit**

```bash
git add scripts/lq/model scripts/lq/train.py scripts/lq/analyze_checkpoint.py
git commit -m "refactor: split lq distnet internals into focused modules"
```

---

### Task 4: Stabilize compatibility and trim import fragility

**Files:**
- Modify: `scripts/lq/train.py`
- Modify: `scripts/lq/datasets.py`
- Modify: `scripts/lq/model/network.py`
- Modify: `scripts/lq/analyze_checkpoint.py`
- Modify: any new package `__init__.py` files as needed

- [ ] **Step 1: Audit entrypoint imports**

Ensure each public entrypoint still works when executed directly with:

- `python scripts/lq/train.py ...`
- `python scripts/lq/analyze_checkpoint.py ...`

without relying on brittle nested fallback chains where a stable package import
would work.

- [ ] **Step 2: Add minimal compatibility re-exports**

If historical imports still reference old module paths, keep shim files thin and
explicit instead of duplicating logic.

- [ ] **Step 3: Verify current shell probes still resolve**

Run:

```bash
bash -n scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh
bash -n scripts/lq/run_train_x_mouth_v20_residual_fsq_sparse_shared_privatecap04_probe.sh
bash -n scripts/lq/run_train_x_mouth_v21_residual_fsq_sparse_shared_privatecap06_probe.sh
```

Expected: no syntax errors.

- [ ] **Step 4: Run final end-to-end smoke**

Run one representative probe command after the refactor:

```bash
bash scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh
```

If 50 epochs is too expensive for the moment, temporarily copy the command into
a one-epoch smoke command and record that the full probe script syntax remains
unchanged.

- [ ] **Step 5: Inspect checkpoint readability**

Run:

```bash
python scripts/lq/analyze_checkpoint.py --checkpoint_path=outputs/lq_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe_win20_e50/best.pt --data_roots=data/win20-step20/IMR,data/win20-step20/TT
```

Expected:

- existing old checkpoint still loads
- summary generation still succeeds

- [ ] **Step 6: Commit**

```bash
git add scripts/lq/train.py scripts/lq/datasets.py scripts/lq/model/network.py scripts/lq/analyze_checkpoint.py scripts/lq/data scripts/lq/training scripts/lq/model
git commit -m "refactor: preserve lq CLI compatibility after module split"
```

---

### Task 5: Update research-facing documentation after the refactor

**Files:**
- Modify: `RESEARCH_PROGRESS.md`
- Modify: `docs/lq_progress.md`
- Modify: `docs/lq_train_presets.md`

- [ ] **Step 1: Add a brief note to progress docs**

Record that the first-round `scripts/lq` refactor:

- preserved current CLI entrypoints
- split training, dataset, and model internals
- did not intentionally change experiment semantics

- [ ] **Step 2: Add path notes where helpful**

If any doc references internal files that moved, update the path references to
the new module locations while noting that compatibility entrypoints remain.

- [ ] **Step 3: Run a doc sanity check**

Run:

```bash
rg -n "scripts/lq/model/network.py|scripts/lq/datasets.py|scripts/lq/train.py" RESEARCH_PROGRESS.md docs/lq_progress.md docs/lq_train_presets.md
```

Expected:

- references that are meant to remain entrypoints still stay
- references to moved internals are corrected where necessary

- [ ] **Step 4: Commit**

```bash
git add RESEARCH_PROGRESS.md docs/lq_progress.md docs/lq_train_presets.md
git commit -m "docs: note lq refactor round1 module split"
```

---

## Final Verification Checklist

- [ ] `python -m py_compile` passes for all new modules.
- [ ] `python scripts/lq/train.py ...` still works.
- [ ] batch memory smoke check still works.
- [ ] one-epoch train smoke still works.
- [ ] `python scripts/lq/analyze_checkpoint.py ...` still works.
- [ ] old checkpoints remain analyzable.
- [ ] current shell probe scripts do not require broad edits.
- [ ] documentation reflects the refactor at a high level.

## Handoff Notes

- Prefer preserving behavior over cleaning up every historical quirk.
- If a refactor choice risks changing current model outputs, stop and add a
  compatibility shim instead.
- Keep each commit scoped to one layer so regressions are easier to isolate.
