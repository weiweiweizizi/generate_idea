# DisentangleNet Label5Class Dual-Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reversible dual-label training path to `scripts/disentangleNet` so `label_5class` can become the default 5-class supervision target without removing the existing 3-class `side` route, while also deepening the shared CNN trunk by inserting one extra `BasicBlock` before each residual downsampling stage.

**Architecture:** Keep the current dataset sample contract intact: both `label_5class` and derived `side_label` remain available in every sample. Extend `DistNet`, training config, and loss wiring so label supervision becomes configurable via `target_label_mode` (`side`, `label5class`, `both`). Update the analysis entrypoints to read checkpoint metadata and choose the correct default label semantics, while still exporting compatibility columns. Deepen the CNN trunk only in `model/encoder.py` and `model/distnet.py`, preserving downstream branch tensor shapes.

**Tech Stack:** Python, PyTorch, Fire CLI, Bash, existing `scripts/disentangleNet` train/analyze workflow, existing `dl` conda environment

---

## File Map

- Modify: `scripts/disentangleNet/train.py`
  - Add explicit label-mode configuration defaults, pass new class counts and label-mode flags into `DistNet`, and update default output naming if needed.
- Modify: `scripts/disentangleNet/training/config.py`
  - Validate the new label-mode options and normalize loss-weight defaults for side vs. label5class supervision.
- Modify: `scripts/disentangleNet/training/losses.py`
  - Route frame/group supervision through `side`, `label_5class`, or both based on config.
- Modify: `scripts/disentangleNet/model/encoder.py`
  - Insert `pre_layer1_block` and `pre_layer2_block` without changing downstream feature sizes.
- Modify: `scripts/disentangleNet/model/heads.py`
  - Add explicit classifier builders for `label5class` frame/group heads.
- Modify: `scripts/disentangleNet/model/distnet.py`
  - Wire the deeper trunk and emit parallel logits for `side` and `label5class`.
- Modify: `scripts/disentangleNet/analysis/analyze_checkpoint.py`
  - Read new checkpoint metadata and default summary/probe semantics correctly.
- Modify: `scripts/disentangleNet/analysis/analyze_side_interpretability.py`
  - Keep side-basis analysis usable on new checkpoints without assuming side is the only label view.
- Modify: `scripts/disentangleNet/analysis/analyze_kfold_report.py`
  - Let the report choose `label_5class` or `side` as its primary grouping target by checkpoint/config.
- Modify: `scripts/disentangleNet/analysis/export_window_basis_activations.py`
  - Ensure both `label_5class` and compatibility `side_label` fields stay in the exported tables.
- Modify: `scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py`
  - Prefer `label_5class` in summaries while preserving laterality rollups.
- Modify: `scripts/disentangleNet/analysis/analyze_patient_tsne.py`
  - Make `label_5class` the default coloring/grouping target for new checkpoints.
- Modify: `scripts/disentangleNet/README.md`
  - Document the new label modes and the updated default analysis perspective.
- Optional Create: `scripts/disentangleNet/run_train_*label5class*.sh`
  - Add a dedicated training preset once the core API stabilizes.

## Pinned References

- Approved spec: `docs/superpowers/specs/2026-05-06-disentanglenet-label5class-dual-path-design.md`
- Current train entrypoint: `scripts/disentangleNet/train.py`
- Current trunk implementation: `scripts/disentangleNet/model/encoder.py`
- Current grouped-loss implementation: `scripts/disentangleNet/training/losses.py`
- Current dataset contract: `scripts/disentangleNet/data/samples.py`

## Constraints

- Do not remove `side_label` or `side_label_name` from dataset samples or exported analysis tables.
- Old checkpoints without `target_label_mode` metadata must still load and be interpreted as side-based runs.
- The deeper trunk must not change the tensor shape expected by `free_adapter`, `side_adapter`, or `private_adapter`.
- Avoid large-scale renaming of `side_semantic_*` basis-path internals in this round; keep the delta focused on supervision and analysis semantics.
- Respect the existing dirty worktree; do not revert unrelated user changes.

## Task 1: Add Config-Level Dual-Label Semantics

**Files:**
- Modify: `scripts/disentangleNet/train.py`
- Modify: `scripts/disentangleNet/training/config.py`
- Test: ad-hoc config smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing config smoke**

```python
from scripts.disentangleNet.train import build_v31_config

cfg = build_v31_config(
    {
        "target_label_mode": "label5class",
        "label5_loss_weight": 0.3,
        "group_label5_loss_weight": 0.7,
    }
)

assert cfg["target_label_mode"] == "label5class"
assert cfg["num_side_classes"] == 3
assert cfg["num_label5_classes"] == 5
assert cfg["label5_loss_weight"] == 0.3
assert cfg["group_label5_loss_weight"] == 0.7
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run:

```bash
python - <<'PY'
from scripts.disentangleNet.train import build_v31_config

cfg = build_v31_config(
    {
        "target_label_mode": "label5class",
        "label5_loss_weight": 0.3,
        "group_label5_loss_weight": 0.7,
    }
)

assert cfg["target_label_mode"] == "label5class"
assert cfg["num_side_classes"] == 3
assert cfg["num_label5_classes"] == 5
assert cfg["label5_loss_weight"] == 0.3
assert cfg["group_label5_loss_weight"] == 0.7
print("dual-label-config-ok")
PY
```

Expected: failure because these config keys are not normalized yet.

- [ ] **Step 3: Implement the config surface**

Add defaults and validation for:

```python
target_label_mode in {"side", "label5class", "both"}
num_side_classes = 3
num_label5_classes = 5
label5_loss_weight
group_label5_loss_weight
```

Ensure `build_v31_config(...)` writes these fields into the saved checkpoint config.

- [ ] **Step 4: Re-run the config smoke**

Run the command from Step 2.

Expected: prints `dual-label-config-ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/disentangleNet/train.py scripts/disentangleNet/training/config.py
git commit -m "disentanglenet: add dual-label config surface"
```

## Task 2: Deepen The Shared CNN Trunk Without Changing Branch Shapes

**Files:**
- Modify: `scripts/disentangleNet/model/encoder.py`
- Modify: `scripts/disentangleNet/model/distnet.py`
- Test: ad-hoc forward-shape smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing trunk-shape smoke**

```python
import torch

from scripts.disentangleNet.model.encoder import build_motion_encoder

modules = build_motion_encoder(hidden_dim=32, pool_size=1)
assert len(modules) == 7

x = torch.randn(2, 1, 119, 119)
feats = modules[0](x)
feats = modules[1](feats)
feats = modules[2](feats)
feats = modules[3](feats)
feats = modules[4](feats)

assert feats.shape[1:] == (32, 15, 15)
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run:

```bash
python - <<'PY'
import torch

from scripts.disentangleNet.model.encoder import build_motion_encoder

modules = build_motion_encoder(hidden_dim=32, pool_size=1)
assert len(modules) == 7

x = torch.randn(2, 1, 119, 119)
feats = modules[0](x)
feats = modules[1](feats)
feats = modules[2](feats)
feats = modules[3](feats)
feats = modules[4](feats)

assert feats.shape[1:] == (32, 15, 15)
print("deeper-trunk-shape-ok")
PY
```

Expected: failure because the encoder does not yet return the two new pre-downsample blocks.

- [ ] **Step 3: Implement the deeper trunk**

In `scripts/disentangleNet/model/encoder.py`, change `build_motion_encoder(...)` so it returns:

```python
initial_conv
pre_layer1_block
layer1
pre_layer2_block
layer2
layer3
avg_pool
```

Use:

- `BasicBlock(8, 8, stride=1)` for `pre_layer1_block`
- `BasicBlock(16, 16, stride=1)` for `pre_layer2_block`

Then update `scripts/disentangleNet/model/distnet.py` to unpack and apply the extra blocks in order.

- [ ] **Step 4: Re-run the trunk-shape smoke**

Run the command from Step 2.

Expected: prints `deeper-trunk-shape-ok`.

- [ ] **Step 5: Run a `DistNet` forward smoke**

Run:

```bash
python - <<'PY'
import torch

from scripts.disentangleNet.model.distnet import DistNet

model = DistNet(
    levels=(2, 6),
    basis_size=119,
    hidden_dim=32,
    pool_size=1,
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
    private_pool_size=1,
    side_semantic_enabled=True,
    side_basis_count=3,
    side_pooling="fixed_region2_contrast",
    quantizer_type="residual_fsq",
    discrete_side_loss_enabled=False,
)

x = torch.randn(2, 4, 1, 119, 119)
side_labels = torch.tensor([0, 2], dtype=torch.long)
out = model(x, side_labels=side_labels)

assert out["reconstructed"].shape == (2, 4, 1, 119, 119)
assert out["side_path_representation"].shape[:2] == (2, 4)
print("distnet-forward-ok")
PY
```

Expected: prints `distnet-forward-ok`.

- [ ] **Step 6: Commit**

```bash
git add scripts/disentangleNet/model/encoder.py scripts/disentangleNet/model/distnet.py
git commit -m "disentanglenet: deepen shared cnn trunk"
```

## Task 3: Add Parallel `label5class` Heads To The Model

**Files:**
- Modify: `scripts/disentangleNet/model/heads.py`
- Modify: `scripts/disentangleNet/model/distnet.py`
- Test: ad-hoc head smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing head-contract smoke**

```python
import torch

from scripts.disentangleNet.model.distnet import DistNet

model = DistNet(
    levels=(2, 6),
    basis_size=119,
    hidden_dim=32,
    pool_size=1,
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
    private_pool_size=1,
    side_semantic_enabled=True,
    side_basis_count=3,
    side_pooling="fixed_region2_contrast",
    quantizer_type="residual_fsq",
    discrete_side_loss_enabled=False,
    num_side_classes=3,
    num_label5_classes=5,
)

x = torch.randn(2, 4, 1, 119, 119)
out = model(x)

assert out["label5_logits"].shape[-1] == 5
assert out["group_label5_logits"].shape[-1] == 5
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run the snippet from Step 1.

Expected: failure because `DistNet` does not yet accept `num_label5_classes` or emit `label5_logits`.

- [ ] **Step 3: Implement the parallel heads**

Add builder helpers in `scripts/disentangleNet/model/heads.py`:

```python
build_label5_classifier(...)
build_group_label5_classifier(...)
```

Then update `DistNet` to:

1. accept `num_label5_classes`
2. instantiate frame/group label5 classifiers
3. emit `label5_logits` and `group_label5_logits` in forward outputs

Keep `side_classifier` and `group_side_classifier` intact.

- [ ] **Step 4: Re-run the head smoke**

Run the snippet from Step 1.

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/disentangleNet/model/heads.py scripts/disentangleNet/model/distnet.py
git commit -m "disentanglenet: add label5class supervision heads"
```

## Task 4: Route Training Losses Through `side`, `label5class`, Or Both

**Files:**
- Modify: `scripts/disentangleNet/training/losses.py`
- Modify: `scripts/disentangleNet/train.py`
- Test: ad-hoc loss-mode smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing loss-mode smoke**

```python
from scripts.disentangleNet.train import build_loss_weights, build_v31_config

cfg = build_v31_config({"target_label_mode": "both"})
weights = build_loss_weights(cfg)

assert "side_group" in weights
assert "label5_group" in weights
assert "label5" in weights
```

- [ ] **Step 2: Run it to verify it fails before implementation**

Run:

```bash
python - <<'PY'
from scripts.disentangleNet.train import build_loss_weights, build_v31_config

cfg = build_v31_config({"target_label_mode": "both"})
weights = build_loss_weights(cfg)

assert "side_group" in weights
assert "label5_group" in weights
assert "label5" in weights
print("dual-loss-mode-ok")
PY
```

Expected: failure because label5 weights are not emitted yet.

- [ ] **Step 3: Implement mode-aware loss routing**

Update `step_model(...)` so it chooses labels and computes losses according to `model.target_label_mode` or config-fed metadata:

- `side`: side-only losses
- `label5class`: label5-only losses
- `both`: both sets summed with their independent weights

Use:

- `batch["side_label"]`
- `batch["label_5class"]`

Keep reconstruction, basis, LQ, and residual losses unchanged.

- [ ] **Step 4: Re-run the loss-mode smoke**

Run the command from Step 2.

Expected: prints `dual-loss-mode-ok`.

- [ ] **Step 5: Run a one-batch train-step smoke**

Run:

```bash
python - <<'PY'
import torch
from torch.optim import AdamW

from scripts.disentangleNet.model.distnet import DistNet
from scripts.disentangleNet.training.losses import step_model

model = DistNet(
    levels=(2, 6),
    basis_size=119,
    hidden_dim=32,
    pool_size=1,
    early_branch_factorization=True,
    free_pool_size=2,
    side_pool_size=2,
    private_pool_size=1,
    side_semantic_enabled=True,
    side_basis_count=3,
    side_pooling="fixed_region2_contrast",
    quantizer_type="residual_fsq",
    discrete_side_loss_enabled=False,
    num_side_classes=3,
    num_label5_classes=5,
    target_label_mode="label5class",
)

batch = {
    "images": torch.randn(2, 4, 1, 119, 119),
    "valid_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
    "padding_mask": torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]], dtype=torch.bool),
    "side_label": torch.tensor([0, 2], dtype=torch.long),
    "label_5class": torch.tensor([1, 4], dtype=torch.long),
}

weights = {
    "recon": 1.0,
    "shared_recon": 1.0,
    "lq": 10.0,
    "orth": 0.1,
    "basis_l1": 1.0,
    "residual": 0.02,
    "subspace_orth": 0.0,
    "side_cont": 0.0,
    "side_disc": 0.0,
    "side_group": 0.0,
    "label5": 0.3,
    "label5_group": 0.7,
}

opt = AdamW(model.parameters(), lr=1e-3)
loss, metrics = step_model(model, batch, "cpu", weights)
loss.backward()
opt.step()

assert "loss" in metrics
print("label5-train-step-ok")
```

Expected: prints `label5-train-step-ok`.

- [ ] **Step 6: Commit**

```bash
git add scripts/disentangleNet/train.py scripts/disentangleNet/training/losses.py
git commit -m "disentanglenet: support dual label supervision modes"
```

## Task 5: Make Analysis Default To The Right Label Semantics

**Files:**
- Modify: `scripts/disentangleNet/analysis/analyze_checkpoint.py`
- Modify: `scripts/disentangleNet/analysis/analyze_kfold_report.py`
- Modify: `scripts/disentangleNet/analysis/analyze_side_interpretability.py`
- Test: ad-hoc checkpoint-config smoke via `python - <<'PY'`

- [ ] **Step 1: Write the failing checkpoint-metadata smoke**

```python
def resolve_primary_label_view(config):
    ...

assert resolve_primary_label_view({}) == "side"
assert resolve_primary_label_view({"target_label_mode": "label5class"}) == "label5class"
assert resolve_primary_label_view({"target_label_mode": "both"}) == "label5class"
```

- [ ] **Step 2: Run it to verify the helper does not exist yet**

Implement a tiny inline import smoke against the chosen helper location after deciding where it should live.

Expected: import or assertion failure before implementation.

- [ ] **Step 3: Implement metadata-aware label-view resolution**

Add one shared helper or equivalent duplicated logic so analysis scripts:

1. default to `side` when checkpoint config lacks new metadata
2. default to `label5class` for new label5 or both-mode runs
3. still allow an explicit CLI override where helpful

Update summaries, probe labels, and human-readable report strings to reflect the selected primary label view.

- [ ] **Step 4: Re-run the metadata smoke**

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/disentangleNet/analysis/analyze_checkpoint.py scripts/disentangleNet/analysis/analyze_kfold_report.py scripts/disentangleNet/analysis/analyze_side_interpretability.py
git commit -m "disentanglenet: make analysis label-view aware"
```

## Task 6: Preserve Compatibility Fields In Activation/Patient Analysis

**Files:**
- Modify: `scripts/disentangleNet/analysis/export_window_basis_activations.py`
- Modify: `scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py`
- Modify: `scripts/disentangleNet/analysis/analyze_patient_tsne.py`
- Modify: `scripts/disentangleNet/README.md`
- Test: export/aggregation smoke on one checkpoint

- [ ] **Step 1: Confirm export paths still include both raw and compatibility labels**

Audit the wide/long row builders and patient summary grouping logic so the exported schema includes:

- `label_5class`
- `side_label`
- `side_label_name`

and does not silently drop side compatibility fields when `label5class` becomes the default view.

- [ ] **Step 2: Implement the schema/summary updates**

Change defaults so:

1. `label_5class` is the preferred grouping/coloring target for new checkpoints
2. side-based rollups remain available as explicit compatibility outputs
3. README examples document both paths clearly

- [ ] **Step 3: Run an activation-export smoke**

Run:

```bash
python scripts/disentangleNet/analysis/export_window_basis_activations.py export \
  --checkpoint_path outputs/disentangleNet/v31_current_verify/best.pt \
  --split all
```

Expected: writes the usual activation CSVs and keeps both `label_5class` and side compatibility columns in the exported schema.

- [ ] **Step 4: Run one patient-level smoke**

Run either:

```bash
python scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py analyze
```

or:

```bash
python scripts/disentangleNet/analysis/analyze_patient_tsne.py analyze \
  --output_root outputs/disentangleNet/v31_current_verify/window_basis_activations_all/patient_pattern_analysis/tsne_smoke
```

Expected: completes without assuming `side_label_name` is the only primary label view.

- [ ] **Step 5: Commit**

```bash
git add scripts/disentangleNet/analysis/export_window_basis_activations.py scripts/disentangleNet/analysis/analyze_patient_activation_patterns.py scripts/disentangleNet/analysis/analyze_patient_tsne.py scripts/disentangleNet/README.md
git commit -m "disentanglenet: preserve side compatibility in label5 analysis"
```

## Task 7: Full Validation Sweep

**Files:**
- Read/produce: one new label5class-mode output directory and associated analysis artifacts
- Optional Create: a dedicated `run_train_...label5class...sh` preset if repeated invocation becomes necessary

- [ ] **Step 1: Run a short smoke training job in `label5class` mode**

Use a small-epoch override against `scripts/disentangleNet/train.py` or a dedicated shell preset. Record:

- whether checkpoint save succeeds
- whether loss keys include label5 metrics
- whether the new deeper trunk trains without shape issues

- [ ] **Step 2: Run a short smoke training job in `side` mode**

Use the same code path with `target_label_mode=side`.

Expected: the previous supervision route still works.

- [ ] **Step 3: Optionally run a `both`-mode smoke**

Only after `side` and `label5class` pass individually.

Expected: dual-head routing works and both losses appear.

- [ ] **Step 4: Run `analyze_checkpoint.py` on the new label5 checkpoint**

Expected: summary reflects `label5class` as the primary view and still exposes side compatibility fields.

- [ ] **Step 5: Write down validation results**

Update a research note or the final PR summary with:

- label mode used
- trunk shape confirmation
- whether old side mode remained functional
- which analysis scripts were rerun successfully

- [ ] **Step 6: Commit or hand off**

If this plan is executed in one branch, keep commits scoped per task above. If validation is the final step in a batch, finish with a summary commit describing the dual-label support and trunk deepening.

