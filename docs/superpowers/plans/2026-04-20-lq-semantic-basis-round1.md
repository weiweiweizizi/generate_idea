# LQ Semantic Basis Round-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-pass `B_side + B_free` shared decomposition to the current `v19` backbone so `side` is supervised through a group-level side-specific shared path without re-enabling dataset auxiliary losses.

**Architecture:** Keep the current `x/mouth` residual-FSQ backbone intact and add a small explicit side semantic bank on the decode side only. The free shared path keeps the existing level-structured basis routing, while the new side path uses a separate compact side basis bank, emits side-specific reconstruction and pooled representations, and receives the only new training supervision in round 1. Dataset disentanglement remains analysis-only in this round.

**Tech Stack:** Python, PyTorch, `vector-quantize-pytorch`, existing `scripts/lq` training/data/analysis stack

---

## File Map

- Modify: `scripts/lq/model/distnet.py`
  - Add side semantic bank parameters, side-path reconstruction, group-level pooled outputs, and a clean forward contract for new analysis fields.
- Modify: `scripts/lq/model/heads.py`
  - Add side-path heads that map `shared_quantized` to side coefficients/weights and group-level side logits.
- Modify: `scripts/lq/training/losses.py`
  - Replace frame-wise side supervision with masked group-level pooling and side loss over pooled side representations.
- Modify: `scripts/lq/train.py`
  - Expose round-1 flags and defaults without re-enabling dataset auxiliary.
- Modify: `scripts/lq/training/config.py`
  - Parse and validate round-1 semantic-basis config knobs.
- Modify: `scripts/lq/analyze_checkpoint.py`
  - Report side/free reconstruction stats and save enough outputs to inspect side-bank usage.
- Modify: `scripts/lq/model/network.py`
  - Keep compatibility export unchanged if `DistNet` signature changes.
- Modify: `scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh`
  - Leave untouched as historical baseline.
- Create: `scripts/lq/run_train_x_mouth_v22_side_semantic_bank_probe.sh`
  - Add a new round-1 training preset that starts from `v19`-like reconstruction hyperparameters and enables side semantic bank options.
- Modify: `RESEARCH_PROGRESS.md`
  - Add a short note after implementation and smoke validation.
- Modify: `docs/lq_progress.md`
  - Record round-1 semantic-bank rationale and observed behavior after runs.

## Constraints

- Do not re-enable `use_dataset_aux` in round 1.
- Do not change dataset file formats or basis init file formats.
- Do not rewrite residual FSQ internals.
- Keep `python scripts/lq/train.py ...` and checkpoint loading stable.
- Keep historical `v19` scripts runnable without new mandatory flags.

## Task 1: Add Round-1 Config Surface

**Files:**
- Modify: `scripts/lq/train.py`
- Modify: `scripts/lq/training/config.py`
- Test: ad-hoc import/CLI validation snippets via `python - <<'PY'`

- [ ] **Step 1: Write the failing config-contract check**

```python
import inspect
from scripts.lq.train import train
from scripts.lq.training.config import prepare_train_config

raw = {
    name: param.default
    for name, param in inspect.signature(train).parameters.items()
}
raw["side_semantic_enabled"] = True
raw["side_basis_count"] = 4
raw["side_pooling"] = "masked_mean"
raw["side_loss_weight"] = 0.3

cfg = prepare_train_config(raw)

assert cfg["side_semantic_enabled"] is True
assert cfg["side_basis_count"] == 4
assert cfg["side_pooling"] == "masked_mean"
assert cfg["side_loss_weight"] == 0.3
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run:

```bash
python - <<'PY'
import inspect
from scripts.lq.train import train
from scripts.lq.training.config import prepare_train_config

raw = {
    name: param.default
    for name, param in inspect.signature(train).parameters.items()
}
raw["side_semantic_enabled"] = True
raw["side_basis_count"] = 4
raw["side_pooling"] = "masked_mean"
raw["side_loss_weight"] = 0.3

cfg = prepare_train_config(raw)

assert cfg["side_semantic_enabled"] is True
assert cfg["side_basis_count"] == 4
assert cfg["side_pooling"] == "masked_mean"
assert cfg["side_loss_weight"] == 0.3
print("config-ok")
PY
```

Expected: `KeyError` for new fields or validation failure before the new flags are added.

- [ ] **Step 3: Implement minimal config support**

Add round-1 flags to `train.py` and `prepare_train_config(...)`:

```python
side_semantic_enabled=False,
side_basis_count=0,
side_pooling="masked_mean",
side_loss_weight=0.0,
```

Validation rules:

```python
if config["side_semantic_enabled"]:
    if int(config["side_basis_count"]) <= 0:
        raise ValueError("side_basis_count must be positive when side_semantic_enabled=True")
    if config["side_pooling"] != "masked_mean":
        raise ValueError("round-1 only supports side_pooling='masked_mean'")
```

Also wire the new training weight in `train.py`:

```python
loss_weights["side_group"] = config["side_loss_weight"]
```

- [ ] **Step 4: Run the config check again**

Run the same `python - <<'PY'` snippet from Step 2.

Expected: prints `config-ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/train.py scripts/lq/training/config.py
git commit -m "lq: add side semantic round1 config flags"
```

## Task 2: Add Side Semantic Bank And Forward Outputs

**Files:**
- Modify: `scripts/lq/model/heads.py`
- Modify: `scripts/lq/model/distnet.py`
- Modify: `scripts/lq/model/network.py` if constructor/export shape changes
- Test: forward-contract smoke script via `python - <<'PY'`

- [ ] **Step 1: Write the failing forward-contract smoke**

```python
import torch
from scripts.lq.model.network import DistNet

model = DistNet(
    levels=(2, 3, 6),
    basis_size=119,
    hidden_dim=32,
    side_semantic_enabled=True,
    side_basis_count=4,
)

x = torch.randn(2, 4, 1, 119, 119)
out = model(x)

assert out["shared_side_reconstruction"].shape == (2, 4, 1, 119, 119)
assert out["shared_free_reconstruction"].shape == (2, 4, 1, 119, 119)
assert out["group_pooled_side_rep"].shape[0] == 2
assert out["side_path_usage"].shape[0] == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
python - <<'PY'
import torch
from scripts.lq.model.network import DistNet

model = DistNet(
    levels=(2, 3, 6),
    basis_size=119,
    hidden_dim=32,
    side_semantic_enabled=True,
    side_basis_count=4,
)

x = torch.randn(2, 4, 1, 119, 119)
out = model(x)

assert out["shared_side_reconstruction"].shape == (2, 4, 1, 119, 119)
assert out["shared_free_reconstruction"].shape == (2, 4, 1, 119, 119)
assert out["group_pooled_side_rep"].shape[0] == 2
assert out["side_path_usage"].shape[0] == 2
print("forward-ok")
PY
```

Expected: constructor error or missing keys before implementation.

- [ ] **Step 3: Implement the side semantic bank minimally**

Add to `DistNet.__init__`:

```python
self.side_semantic_enabled = side_semantic_enabled
self.side_basis_count = side_basis_count
self.side_basis_bank = nn.Parameter(
    torch.randn(side_basis_count, basis_size, basis_size) * 0.02
)
```

Add small side heads in `heads.py`:

```python
def build_side_semantic_coeff_head(shared_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(shared_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, 1),
    )

def build_side_semantic_basis_head(shared_dim, hidden_dim, side_basis_count):
    return nn.Sequential(
        nn.Linear(shared_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, side_basis_count),
    )
```

In `forward(...)`, produce:

```python
shared_side_reconstruction
shared_free_reconstruction
group_pooled_side_rep=None
group_pooled_free_rep=None
side_path_usage
free_path_usage
```

Round-1 rule: rename the current shared path output to `shared_free_reconstruction`, then define:

```python
shared_reconstruction = shared_side_reconstruction + shared_free_reconstruction
```

Do not leave the old shared path as “full old shared + extra side add-on”; round 1 must implement a real side/free split in the shared reconstruction contract.

- [ ] **Step 4: Run the forward-contract smoke again**

Run the same `python - <<'PY'` snippet from Step 2.

Expected: prints `forward-ok`.

- [ ] **Step 5: Run compile smoke**

```bash
python -m py_compile scripts/lq/model/heads.py scripts/lq/model/distnet.py scripts/lq/model/network.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add scripts/lq/model/heads.py scripts/lq/model/distnet.py scripts/lq/model/network.py
git commit -m "lq: add side semantic bank forward path"
```

## Task 3: Replace Frame-Wise Side Loss With Group-Level Pooling

**Files:**
- Modify: `scripts/lq/training/losses.py`
- Modify: `scripts/lq/model/distnet.py`
- Test: targeted batch-step smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing group-level side-loss check**

```python
import torch
from scripts.lq.training.losses import masked_mean

values = torch.tensor([[1.0, 3.0], [2.0, 8.0]])
mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

pooled = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
assert torch.allclose(pooled, torch.tensor([2.0, 2.0]))
```

Then require `step_model(...)` to expose a `side_group` metric sourced from pooled reps, not frame-wise logits.

- [ ] **Step 2: Run a failing batch-step smoke**

Run:

```bash
python - <<'PY'
import torch
from scripts.lq.model.network import DistNet
from scripts.lq.training.losses import step_model

model = DistNet(
    levels=(2, 3, 6),
    basis_size=119,
    hidden_dim=16,
    side_semantic_enabled=True,
    side_basis_count=2,
)

batch = {
    "images": torch.randn(2, 4, 1, 119, 119),
    "valid_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
    "padding_mask": torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=torch.bool),
    "side_label": torch.tensor([0, 2]),
    "dataset_label": torch.tensor([0, 1]),
}

loss, metrics = step_model(
    model,
    batch,
    device="cpu",
    loss_weights={
        "recon": 1.0,
        "shared_recon": 0.0,
        "lq": 1.0,
        "orth": 0.0,
        "basis_l1": 0.0,
        "residual": 0.0,
        "side_cont": 0.0,
        "side_disc": 0.0,
        "side_group": 1.0,
        "dataset_private": 0.0,
        "dataset_adv": 0.0,
    },
)

assert "side_group" in metrics
print("group-loss-ok")
PY
```

Expected: unknown weight key or missing metric before implementation.

- [ ] **Step 3: Implement pooled side supervision**

Add pooled helper in `losses.py`:

```python
def masked_mean_per_sequence(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.device, values.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denom
```

Use model outputs:

```python
group_pooled_side_rep = outputs["group_pooled_side_rep"]
group_side_logits = outputs["group_side_logits"]
```

Compute:

```python
side_group_loss = F.cross_entropy(group_side_logits, batch["side_label"].to(device))
```

Round-1 rule:
- `side_cont` and `side_disc` stay defaulted to `0.0`
- do not add them into `total_loss` when `side_semantic_enabled=True`
- use `loss_weights["side_group"]` as the only active side supervision term in round 1

- [ ] **Step 4: Run the batch-step smoke again**

Run the same `python - <<'PY'` snippet from Step 2.

Expected: prints `group-loss-ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/model/distnet.py scripts/lq/training/losses.py
git commit -m "lq: switch side supervision to grouped semantic loss"
```

## Task 4: Extend Analysis Outputs For Side vs Free Paths And Probes

**Files:**
- Modify: `scripts/lq/analyze_checkpoint.py`
- Test: checkpoint-analysis smoke against a tiny round-1 run or direct forward dump

- [ ] **Step 1: Write the expected analysis fields**

Document and assert that round-1 analysis emits:

```python
required_keys = [
    "shared_side_reconstruction",
    "shared_free_reconstruction",
    "side_path_usage",
    "free_path_usage",
    "side_probe",
    "dataset_probe",
]
```

- [ ] **Step 2: Run current analysis flow to verify fields are missing**

Run:

```bash
python scripts/lq/analyze_checkpoint.py --help
```

Then run analysis on the first round-1 checkpoint once it exists and confirm the new outputs are absent before implementation.

- [ ] **Step 3: Implement minimal round-1 analysis**

Add summary items for:

```python
{
    "mean_side_path_usage": ...,
    "mean_free_path_usage": ...,
    "mean_side_recon_l1": ...,
    "mean_free_recon_l1": ...,
    "side_probe": {
        "side_from_side_rep_acc": ...,
        "side_from_free_rep_acc": ...,
    },
    "dataset_probe": {
        "dataset_from_side_rep_acc": ...,
        "dataset_from_free_rep_acc": ...,
        "dataset_from_private_rep_acc": ...,
    },
}
```

Implementation target:

- dump `group_pooled_side_rep`, `group_pooled_free_rep`, `group_pooled_private_rep` plus labels into an analysis artifact
- fit a lightweight probe inside analysis (prefer `sklearn.linear_model.LogisticRegression` if already available in env; otherwise document and implement a simple linear probe fallback)
- save side/free basis visualizations and probe summaries into the checkpoint `analysis/` directory

- [ ] **Step 4: Run analysis smoke**

Run:

```bash
python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v22_side_semantic_bank_probe/best.pt \
  --output_dir outputs/lq_x_mouth_v22_side_semantic_bank_probe/analysis
```

Expected: exits successfully and writes updated analysis summaries.

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/analyze_checkpoint.py
git commit -m "lq: add side free path analysis outputs"
```

## Task 5: Add Round-1 Training Preset And Smoke Validation

**Files:**
- Create: `scripts/lq/run_train_x_mouth_v22_side_semantic_bank_probe.sh`
- Modify: `docs/lq_progress.md`
- Modify: `RESEARCH_PROGRESS.md`
- Test: one-epoch smoke run and analysis smoke

- [ ] **Step 1: Write the preset script**

Create:

```bash
#!/usr/bin/env bash
python scripts/lq/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --group_size=4 \
  --mode=x \
  --region=mouth \
  --basis_size=119 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x.npy \
  --hidden_dim=32 \
  --private_dim=32 \
  --pool_size=1 \
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
  --epochs=1 \
  --batch_size=64 \
  --side_semantic_enabled=True \
  --side_basis_count=2 \
  --side_loss_weight=0.3 \
  --side_cont_weight=0.0 \
  --side_disc_weight=0.0 \
  --use_dataset_aux=False \
  --output_dir=outputs/lq_x_mouth_v22_side_semantic_bank_probe_smoke
```

- [ ] **Step 2: Run shell syntax check**

```bash
bash -n scripts/lq/run_train_x_mouth_v22_side_semantic_bank_probe.sh
```

Expected: no output.

- [ ] **Step 3: Run a 1-epoch smoke**

```bash
python scripts/lq/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --group_size=4 \
  --mode=x \
  --region=mouth \
  --basis_size=119 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x.npy \
  --hidden_dim=32 \
  --private_dim=32 \
  --pool_size=1 \
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
  --epochs=1 \
  --batch_size=64 \
  --quantizer_type=residual_fsq \
  --fsq_preserve_symmetry=True \
  --basis_orthogonalization=global_qr \
  --side_semantic_enabled=True \
  --side_basis_count=2 \
  --side_loss_weight=0.3 \
  --side_cont_weight=0.0 \
  --side_disc_weight=0.0 \
  --use_dataset_aux=False \
  --output_dir=outputs/lq_x_mouth_v22_side_semantic_bank_probe_smoke
```

Expected:
- training completes
- `best.pt` is written
- logs contain `side_group` metric

- [ ] **Step 4: Run checkpoint analysis smoke**

```bash
python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v22_side_semantic_bank_probe_smoke/best.pt
```

Expected: analysis completes and writes side/free summaries.

- [ ] **Step 5: Update research notes**

Append a short round-1 note to:
- `RESEARCH_PROGRESS.md`
- `docs/lq_progress.md`

Include:
- whether `B_side` was used
- whether side probe improved
- whether free path still appeared to carry side signal

- [ ] **Step 6: Commit**

```bash
git add \
  scripts/lq/run_train_x_mouth_v22_side_semantic_bank_probe.sh \
  RESEARCH_PROGRESS.md \
  docs/lq_progress.md
git commit -m "lq: add side semantic bank round1 preset"
```

## Task 6: Final Regression Sweep

**Files:**
- Modify: none unless regressions are found
- Test: import/compile/smoke commands

- [ ] **Step 1: Run import regression**

```bash
python - <<'PY'
import scripts.lq.train
import scripts.lq.datasets
import scripts.lq.model.network
import scripts.lq.analyze_checkpoint
print("imports-ok")
PY
```

Expected: prints `imports-ok`.

- [ ] **Step 2: Run compile regression**

```bash
python -m py_compile \
  scripts/lq/train.py \
  scripts/lq/training/config.py \
  scripts/lq/training/losses.py \
  scripts/lq/model/heads.py \
  scripts/lq/model/distnet.py \
  scripts/lq/model/network.py \
  scripts/lq/analyze_checkpoint.py
```

Expected: no output.

- [ ] **Step 3: Re-run historical baseline shell syntax check**

```bash
bash -n scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh
```

Expected: no output.

- [ ] **Step 4: Run old-checkpoint compatibility smoke**

Run:

```bash
python scripts/lq/analyze_checkpoint.py \
  outputs/lq_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe_win20_e50/best.pt \
  --output_dir outputs/lq_round1_old_ckpt_compat_analysis
```

Expected: analysis completes successfully on a pre-round1 checkpoint without requiring side-bank weights.

- [ ] **Step 5: Commit or document any final fixups**

```bash
git status --short
```

Expected: either clean worktree for plan execution branch, or only intentional follow-up edits.
