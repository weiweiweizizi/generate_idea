# scripts/lq Refactor Design

Last updated: 2026-04-19

## Goal

Refactor `scripts/lq` to improve maintainability and readability without
changing the current experiment semantics, CLI entrypoints, checkpoint format,
or probe workflow in any material way.

This is a structural cleanup pass for the active LQ prototype, not a training
behavior redesign.

## Context

The current `scripts/lq` subtree has accumulated multiple experiment rounds.
Three files now carry too many responsibilities:

- `scripts/lq/model/network.py`
- `scripts/lq/datasets.py`
- `scripts/lq/train.py`

The immediate problems are:

- files are too long to reason about safely
- model logic, configuration, loss computation, IO, and orchestration are mixed
- future probes are likely to add even more conditionals into already-large
  files
- analysis and training rely on implicit coupling instead of clear boundaries

At the same time, the current code is actively used for ongoing experiments, so
the refactor must preserve existing workflows.

## Non-Goals

This first refactor round will not:

- redesign the training objective
- change current loss definitions or output semantics on purpose
- rewrite the historical `run_train_x_mouth_v*.sh` probe scripts
- standardize every old experiment artifact
- change the checkpoint payload structure beyond internal refactoring needs
- introduce a new framework or trainer abstraction

## Constraints

The refactor must preserve the following external behavior as much as possible:

- `python scripts/lq/train.py ...` remains the main training entrypoint
- `python scripts/lq/analyze_checkpoint.py ...` remains the checkpoint analysis
  entrypoint
- current dataset batch keys stay stable
- current `DistNet` output keys stay stable
- current checkpoint contents remain readable by existing analysis code
- current shell scripts under `scripts/lq/run_train_x_mouth_v*.sh` should keep
  working without broad edits

The refactor may clean up internal structure and remove obvious bad smells as
long as these compatibility constraints remain satisfied.

## Recommended Approach

Use a medium-strength refactor:

- keep current top-level CLI files and core public names
- split the implementation into focused internal modules
- move logic out of oversized files in stages
- preserve behavior first, then clean up repetitive or tangled internals while
  touching them

This is preferred over a minimal split because the current boundaries are too
weak, and preferred over a full rewrite because the code is still in active
research use.

## Target Structure

### `scripts/lq/data/`

Responsibility: dataset metadata, subject splitting, matrix loading,
normalization helpers, grouping logic, and stable sample construction.

Planned files:

- `specs.py`
  - `DatasetSpec`
  - label helpers
  - subject split helpers
- `io.py`
  - metadata loading
  - matrix loading
  - subject width inference
  - global signed scale estimation
- `samples.py`
  - common sample-dict construction
  - masks and source semantics
- `datasets.py`
  - `FacialMotionDataset`
  - `FacialMotionSequenceDataset`

### `scripts/lq/model/`

Responsibility: compose the LQ model from smaller parts while preserving the
public `DistNet` API.

Planned files:

- `encoder.py`
  - CNN stem
  - residual encoder blocks
- `basis.py`
  - basis init loading
  - matrix constraints
  - basis normalization / QR projection
  - basis penalties
- `quantizers.py`
  - unified wrapper for `LatentQuantize`, `FSQ`, and residual-FSQ behavior
- `heads.py`
  - shared head
  - private head
  - side supervision heads
  - dataset auxiliary heads
- `distnet.py`
  - `DistNet`
  - forward pass orchestration
- `network.py`
  - compatibility wrapper that re-exports `DistNet`

### `scripts/lq/training/`

Responsibility: training configuration, dataset construction, step logic, epoch
execution, memory smoke checks, and checkpoint saving.

Planned files:

- `config.py`
  - argument defaults
  - validation helpers
- `data.py`
  - dataset spec construction
  - train/val dataset building
  - dataloader helpers
- `losses.py`
  - masked reductions
  - batch step computation
- `engine.py`
  - epoch loop
  - batch memory validation
- `checkpoint.py`
  - best-checkpoint saving helper

### `scripts/lq/analysis/`

Responsibility: checkpoint inspection, basis visualization, and code-usage
summary helpers.

Planned files:

- `checkpoint_analysis.py`
  - analysis orchestration now living in `analyze_checkpoint.py`
- `visualize.py`
  - basis heatmap rendering
  - code-usage summary helpers

### Entry Files Kept Stable

- `scripts/lq/train.py`
- `scripts/lq/analyze_checkpoint.py`

These files should become thin entrypoints that delegate to the new modules.

## Refactor Order

### Phase 1: Training Layer

First split `scripts/lq/train.py`.

Reason:

- lowest risk compared with model internals
- immediate readability gain
- creates cleaner hooks for later smoke tests and future probes

Expected result:

- `train.py` becomes mostly CLI argument entry logic
- dataset build, step logic, and epoch running are moved into
  `scripts/lq/training/`

### Phase 2: Dataset Layer

Then split `scripts/lq/datasets.py`.

Reason:

- current file mixes metadata access, sample semantics, sequence grouping, and
  normalization
- dataset behavior must remain stable, so it is safer to do this after the
  training loop has clearer seams

Expected result:

- metadata and matrix IO move out of dataset classes
- grouped-sequence behavior remains unchanged
- output fields remain stable

### Phase 3: Model Layer

Finally split `scripts/lq/model/network.py`.

Reason:

- highest risk area
- contains basis logic, quantization logic, supervision heads, sequence reshape
  helpers, and forward orchestration all at once
- easier to split safely after training and dataset boundaries are already
  cleaned up

Expected result:

- `DistNet` remains the public model class
- basis, quantizer, encoder, and auxiliary heads become focused modules
- `network.py` remains as compatibility shim

## Compatibility Rules

The following compatibility rules apply to the first refactor pass:

- keep `DistNet` constructor arguments compatible unless a bug forces a change
- keep existing model output keys unchanged
- keep checkpoint `config`, `train_metrics`, and `val_metrics` structure
  compatible
- keep analysis inputs compatible with existing saved checkpoints
- keep dataset sample fields identical unless the change is strictly internal

Where a structural move would otherwise break imports, prefer lightweight
compatibility wrappers over sweeping callsite changes.

## Allowed Cleanup While Refactoring

The refactor may also correct local maintainability issues encountered on the
way, as long as semantics stay aligned:

- replace repeated code with shared helpers
- move validation checks closer to configuration boundaries
- shrink oversized functions into smaller named units
- simplify fragile import fallback chains when a cleaner package-relative import
  can be made reliable
- add concise comments where control flow is non-obvious

The refactor should not opportunistically redesign the research logic.

## Validation Plan

The first refactor round is considered successful if all of the following pass:

1. Training entrypoint still runs:
   - `python scripts/lq/train.py ...`
2. Batch memory smoke check still runs successfully.
3. A lightweight train smoke run reaches at least the memory check and one
   epoch without interface regressions.
4. Checkpoint analysis still runs:
   - `python scripts/lq/analyze_checkpoint.py ...`
5. Existing saved checkpoints remain analyzable.
6. Dataset batch keys and model output keys remain stable.

## Implementation Boundary for Round 1

Round 1 should stop after the core structure is improved for:

- `train.py`
- `datasets.py`
- `model/network.py`

It is acceptable to leave the historical shell probe scripts and smaller helper
files mostly unchanged, as long as they continue to work with the refactored
core.

## Expected Outcome

After this refactor:

- each major concern in `scripts/lq` should have a clearer home
- future probe additions should require smaller, more local edits
- the active research workflow should remain usable without retraining users on
  a new interface
- subsequent structural work should become safer because the code will be
  decomposed into smaller units
