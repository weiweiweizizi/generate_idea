# LQ Early Branch Factorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the late `shared_quantized -> split side/free` factorization with an early branch architecture where `free` and `side` diverge at the feature-map stage, while keeping the current single-direction `x/mouth` training pipeline runnable end-to-end.

**Architecture:** Keep the current CNN trunk and residual-FSQ free-motion decoder, but insert three lightweight branch adapters on top of `layer3` features. Each branch gets its own pooling and head, only `free_z` is quantized, and `side_z` is consumed directly by the side basis / side supervision path. Backward compatibility is preserved at the CLI and checkpoint-loading level where reasonable, but the new baseline must stop depending on split slices of `shared_quantized`.

**Tech Stack:** Python, PyTorch, `vector-quantize-pytorch`, existing `scripts/lq` train/analyze workflow, `dl` conda environment

---

## File Map

- Modify: `scripts/lq/model/encoder.py`
  - Add branch-adapter builder utilities and branch-specific pooling support without changing the existing trunk contract.
- Modify: `scripts/lq/model/heads.py`
  - Add `free_head` / `side_head` builders if needed, keep head interfaces explicit, and avoid reusing the old split-latent assumptions.
- Modify: `scripts/lq/model/distnet.py`
  - Replace late split logic with early branch adapters, dedicated pooling, new forward outputs, and a branch-specific reconstruction contract.
- Modify: `scripts/lq/model/network.py`
  - Keep the public import/export path for `DistNet` stable if signatures move.
- Modify: `scripts/lq/training/config.py`
  - Add early-branch config surface and validation defaults for pool sizes and branch dimensions.
- Modify: `scripts/lq/train.py`
  - Expose the new config knobs and keep the default training command aligned with the new baseline assumptions.
- Modify: `scripts/lq/training/losses.py`
  - Remove dependence on split side/free latent tensors, treat old QR/subspace/adversarial items as disabled by default, and keep loss reporting stable.
- Modify: `scripts/lq/analyze_checkpoint.py`
  - Read the new branch outputs, compute probes from true `side_z/free_z`, and remain backward-compatible with older checkpoints.
- Create: `scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh`
  - Add a dedicated preset for the new baseline on `data/win20-step20/IMR,data/win20-step20/TT`.
- Modify: `RESEARCH_PROGRESS.md`
  - Add a short summary after smoke validation and after the first full run.

## Constraints

- Every Python command must run under `dl`:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python ...
```

- Do not change dataset formats or basis init formats.
- Do not rewrite residual FSQ internals.
- Do not mix this structural change with new adversarial losses, QR-on-z, or attention pooling in the first pass.
- Keep old historical checkpoints analyzable.
- Preserve current `train.py` usage on grouped sequence input `(B, T, 1, H, W)`.

## Task 1: Add Early-Branch Config Surface

**Files:**
- Modify: `scripts/lq/train.py`
- Modify: `scripts/lq/training/config.py`
- Test: ad-hoc config-contract snippet via `python - <<'PY'`

- [ ] **Step 1: Write the failing config-contract check**

```python
import inspect
from scripts.lq.train import train
from scripts.lq.training.config import prepare_train_config

raw = {
    name: param.default
    for name, param in inspect.signature(train).parameters.items()
}
raw["early_branch_factorization"] = True
raw["free_pool_size"] = 2
raw["side_pool_size"] = 2
raw["private_pool_size"] = 1
raw["free_z_dim"] = 32
raw["side_z_dim"] = 32

cfg = prepare_train_config(raw)

assert cfg["early_branch_factorization"] is True
assert cfg["free_pool_size"] == 2
assert cfg["side_pool_size"] == 2
assert cfg["private_pool_size"] == 1
assert cfg["free_z_dim"] == 32
assert cfg["side_z_dim"] == 32
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
import inspect
from scripts.lq.train import train
from scripts.lq.training.config import prepare_train_config

raw = {
    name: param.default
    for name, param in inspect.signature(train).parameters.items()
}
raw["early_branch_factorization"] = True
raw["free_pool_size"] = 2
raw["side_pool_size"] = 2
raw["private_pool_size"] = 1
raw["free_z_dim"] = 32
raw["side_z_dim"] = 32

cfg = prepare_train_config(raw)

assert cfg["early_branch_factorization"] is True
assert cfg["free_pool_size"] == 2
assert cfg["side_pool_size"] == 2
assert cfg["private_pool_size"] == 1
assert cfg["free_z_dim"] == 32
assert cfg["side_z_dim"] == 32
print("config-ok")
PY
```

Expected: missing-key or validation failure before the new config exists.

- [ ] **Step 3: Implement the minimal config surface**

Add these defaults to `train.py`:

```python
early_branch_factorization=False,
free_pool_size=2,
side_pool_size=2,
private_pool_size=1,
free_z_dim=None,
side_z_dim=None,
private_adapter_enabled=False,
```

Validation rules in `prepare_train_config(...)`:

```python
if config["early_branch_factorization"]:
    config.setdefault("free_z_dim", config["hidden_dim"])
    config.setdefault("side_z_dim", config["hidden_dim"])
    if int(config["free_pool_size"]) <= 0 or int(config["side_pool_size"]) <= 0:
        raise ValueError("branch pool sizes must be positive")
    if int(config["free_z_dim"]) <= 0 or int(config["side_z_dim"]) <= 0:
        raise ValueError("branch latent dims must be positive")
```

Compatibility rule:

```python
if config["early_branch_factorization"]:
    config["side_free_frame_qr"] = False
    config["free_side_adv_weight"] = 0.0
    config["subspace_orth_weight"] = 0.0
```

- [ ] **Step 4: Run the config-contract check again**

Run the same command from Step 2.

Expected: prints `config-ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/train.py scripts/lq/training/config.py
git commit -m "lq: add early branch factorization config"
```

## Task 2: Add Branch Adapters And Heads

**Files:**
- Modify: `scripts/lq/model/encoder.py`
- Modify: `scripts/lq/model/heads.py`
- Modify: `scripts/lq/model/network.py`
- Test: builder/import smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing branch-builder smoke**

```python
from scripts.lq.model.encoder import build_branch_adapter, build_branch_pool
from scripts.lq.model.heads import build_free_head, build_side_head

adapter = build_branch_adapter(32)
pool = build_branch_pool(2)
free_head = build_free_head(32 * 2 * 2, 32, 32)
side_head = build_side_head(32 * 2 * 2, 32, 32)

assert adapter is not None
assert pool.output_size == (2, 2)
assert free_head is not None
assert side_head is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python - <<'PY'
from scripts.lq.model.encoder import build_branch_adapter, build_branch_pool
from scripts.lq.model.heads import build_free_head, build_side_head

adapter = build_branch_adapter(32)
pool = build_branch_pool(2)
free_head = build_free_head(32 * 2 * 2, 32, 32)
side_head = build_side_head(32 * 2 * 2, 32, 32)

assert adapter is not None
assert pool.output_size == (2, 2)
assert free_head is not None
assert side_head is not None
print("branch-build-ok")
PY
```

Expected: import error or missing builder failure before the new branch utilities are added.

- [ ] **Step 3: Implement branch-specific builders**

In `encoder.py`, add focused helpers instead of hardcoding branch logic inside `DistNet`:

```python
def build_branch_adapter(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=True),
    )

def build_branch_pool(pool_size: int) -> nn.AdaptiveAvgPool2d:
    return nn.AdaptiveAvgPool2d((pool_size, pool_size))
```

In `heads.py`, add explicit branch heads:

```python
def build_free_head(pooled_dim: int, hidden_dim: int, free_z_dim: int) -> nn.Sequential: ...
def build_side_head(pooled_dim: int, hidden_dim: int, side_z_dim: int) -> nn.Sequential: ...
```

Keep `build_private_head(...)` reusable for the private path.

- [ ] **Step 4: Run the branch-builder smoke again**

Run the same command from Step 2.

Expected: prints `branch-build-ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/model/encoder.py scripts/lq/model/heads.py scripts/lq/model/network.py
git commit -m "lq: add branch adapters and heads"
```

## Task 3: Rewrite `DistNet` Forward To Early Factorization

**Files:**
- Modify: `scripts/lq/model/distnet.py`
- Modify: `scripts/lq/model/network.py` if export glue changes
- Test: forward-contract smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing forward-contract check**

```python
import torch
from scripts.lq.model.network import DistNet

model = DistNet(
    levels=(2, 3, 6),
    basis_size=119,
    hidden_dim=32,
    private_dim=32,
    side_semantic_enabled=True,
    side_basis_count=2,
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
)

x = torch.randn(2, 4, 1, 119, 119)
side_labels = torch.tensor([0, 1])
out = model(x, side_labels=side_labels, return_group_pooled=True)

assert out["shared_free_reconstruction"].shape == (2, 4, 1, 119, 119)
assert out["shared_side_reconstruction"].shape == (2, 4, 1, 119, 119)
assert out["group_pooled_free_rep"].shape == (2, 32)
assert out["group_pooled_side_rep"].shape == (2, 32)
assert "side_latent_raw" not in out or out["side_latent_raw"] is None
print("forward-ok")
```

- [ ] **Step 2: Run it to verify it fails**

Run the same snippet under `dl`.

Expected: missing outputs or old split-based outputs still drive the contract.

- [ ] **Step 3: Implement the forward rewrite minimally**

Branch logic should look like this:

```python
feats = trunk(x)

free_feats = self.free_adapter(feats)
side_feats = self.side_adapter(feats)
private_feats = self.private_adapter(feats) if self.private_adapter is not None else feats

free_pooled = self.free_pool(free_feats).flatten(1)
side_pooled = self.side_pool(side_feats).flatten(1)
private_pooled = self.private_pool(private_feats).flatten(1)

free_raw = self.free_head(free_pooled)
side_z = self.side_head(side_pooled)
private_z = self.private_head(private_pooled)

free_quantized, indices, stage_quantized = self._quantize_shared(free_raw)
```

Decode rules:

- use `free_quantized` for shared coeff heads and shared basis heads
- use `side_z` for side semantic coeff/basis heads and side classifier
- do not call `_split_side_free_latent(...)` in early-branch mode
- do not call `_orthogonalize_side_free_latent(...)` in early-branch mode

Output contract in early-branch mode:

```python
{
    "free_latent": ...,
    "side_latent": ...,
    "group_pooled_free_rep": ...,
    "group_pooled_side_rep": ...,
    "shared_free_reconstruction": ...,
    "shared_side_reconstruction": ...,
}
```

Backward-compatibility rule:

- keep old fields available as `None` or alias only if analysis code still needs them
- do not fake split-derived tensors from `free_quantized`
- treat `group_pooled_free_rep` / `group_pooled_side_rep` as the canonical pooled branch outputs
- if useful for diagnostics, `group_pooled_*_latent` may be emitted as aliases, but analysis and probes must key off `*_rep`

- [ ] **Step 4: Run the forward-contract smoke again**

Run the same command from Step 2.

Expected: prints `forward-ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/model/distnet.py scripts/lq/model/network.py
git commit -m "lq: switch distnet to early branch factorization"
```

## Task 4: Update Loss Assembly And Analysis

**Files:**
- Modify: `scripts/lq/training/losses.py`
- Modify: `scripts/lq/analyze_checkpoint.py`
- Test: loss smoke + analysis smoke via `python - <<'PY'` and `analyze_checkpoint.py`

- [ ] **Step 1: Write the failing loss/analysis contract check**

```python
import torch
from scripts.lq.model.network import DistNet
from scripts.lq.training.losses import compute_losses

model = DistNet(
    levels=(2, 3, 6),
    basis_size=119,
    hidden_dim=32,
    private_dim=32,
    side_semantic_enabled=True,
    side_basis_count=2,
    early_branch_factorization=True,
)
x = torch.randn(2, 4, 1, 119, 119)
side_labels = torch.tensor([0, 1])
out = model(x, side_labels=side_labels, return_group_pooled=True)

loss_dict = compute_losses(
    outputs=out,
    batch={"image": x, "images": x, "side_label": side_labels},
    loss_weights={"recon": 1.0, "shared_recon": 1.0, "side_group": 0.3},
)

assert "subspace_orth" in loss_dict
assert torch.isfinite(loss_dict["loss"])
assert out["group_pooled_free_rep"] is not None
assert out["group_pooled_side_rep"] is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run the snippet under `dl`.

Expected: loss code still expects split-based tensors or analysis code cannot read new branch outputs.

- [ ] **Step 3: Implement loss and analysis updates**

Loss rules:

- keep `recon`, `shared_recon`, `lq`, `orth`, `basis_l1`, `residual`, `side_group`
- in early-branch mode, force:

```python
subspace_orth = outputs["reconstructed"].new_zeros(())
free_side_adv = outputs["reconstructed"].new_zeros(())
```

- do not compute latent orthogonality from old split slices

Analysis rules:

- probe `group_pooled_side_rep` and `group_pooled_free_rep` if present
- require `group_pooled_side_rep` and `group_pooled_free_rep` for early-branch checkpoints
- only use fallback paths for old checkpoints that predate early-branch factorization
- compute latent linear recoverability on true `side_latent/free_latent`
- keep older checkpoints working by branching on key existence

- [ ] **Step 4: Run analysis smoke on a 1-epoch checkpoint**

First train a 1-epoch checkpoint:

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
  --epochs=1 \
  --batch_size=64 \
  --side_semantic_enabled=True \
  --side_basis_count=2 \
  --side_loss_weight=0.3 \
  --quantizer_type=residual_fsq \
  --basis_orthogonalization=global_qr \
  --shared_basis_soft_mixing=True \
  --shared_basis_anchor_bias=2.0 \
  --shared_basis_topk=2 \
  --early_branch_factorization=True \
  --free_pool_size=2 \
  --side_pool_size=2 \
  --output_dir=outputs/lq_x_mouth_v26_early_branch_smoke
```

Then analyze:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  --checkpoint_path=outputs/lq_x_mouth_v26_early_branch_smoke/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT
```

Expected:

- checkpoint saves successfully
- analysis completes
- summary contains side/free latent or group-representation probe fields

- [ ] **Step 5: Run backward-compat analysis smoke on an older checkpoint**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  --checkpoint_path=outputs/lq_x_mouth_v23_side_subspace_orth_probe_win20_e50/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT
```

Expected:

- analysis completes without requiring new branch keys
- summary is regenerated or updated successfully
- no crash caused by the early-branch output contract

- [ ] **Step 6: Commit**

```bash
git add scripts/lq/training/losses.py scripts/lq/analyze_checkpoint.py
git commit -m "lq: update losses and analysis for early branches"
```

## Task 5: Add Baseline Preset And Run Validation

**Files:**
- Create: `scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh`
- Modify: `RESEARCH_PROGRESS.md`
- Test: 1-epoch smoke and 50-epoch full run

- [ ] **Step 1: Add the dedicated preset script**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/weizilin/anaconda3/etc/profile.d/conda.sh
conda activate dl

python scripts/lq/train.py \
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
  --epochs=50 \
  --batch_size=64 \
  --side_semantic_enabled=True \
  --side_basis_count=2 \
  --side_loss_weight=0.3 \
  --use_dataset_aux=False \
  --early_branch_factorization=True \
  --free_pool_size=2 \
  --side_pool_size=2 \
  --private_pool_size=1 \
  --output_dir=outputs/lq_x_mouth_v26_early_branch_probe_win20_e50
```

- [ ] **Step 2: Run the preset for 1 epoch first**

Temporarily override `--epochs=1` inline or duplicate the smoke command from Task 4 Step 4.

Expected:

- no shape mismatches
- batch-memory validation passes
- best checkpoint and analysis are generated

- [ ] **Step 3: Run the full 50-epoch baseline**

Run:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && bash scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh
```

Then analyze:

```bash
source /home/weizilin/anaconda3/etc/profile.d/conda.sh && conda activate dl && python scripts/lq/analyze_checkpoint.py \
  --checkpoint_path=outputs/lq_x_mouth_v26_early_branch_probe_win20_e50/best.pt \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT
```

Expected go/no-go checks:

- `side_from_side_rep > side_from_free_rep`
- `dataset_from_side_rep <= 0.7318`
- `raw_linear_r2_free_to_side < 0.95`
- `val_recon <= 0.3543`

- [ ] **Step 4: Record the outcome**

Add a short note to `RESEARCH_PROGRESS.md`:

- command used
- best epoch
- key validation metrics
- whether the early-branch hypothesis passed or failed

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/run_train_x_mouth_v26_early_branch_probe.sh RESEARCH_PROGRESS.md
git commit -m "lq: add early branch baseline preset"
```
