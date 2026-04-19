# LQ FSQ Baseline Next Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `v10 FSQ` to the new `scripts/lq` baseline, then run the next round of structure and supervision probes on top of it in a controlled order.

**Architecture:** Keep the current single-direction `x + mouth` pipeline as the canonical comparison setup, with `FSQ` as the default shared quantizer. Separate the next phase into four layers: baseline freeze, baseline analysis, FSQ-based structure probes, and cross-setting validation. Every new probe must be compared against `v10` with the same data roots and reporting format.

**Tech Stack:** Python, PyTorch, `vector-quantize-pytorch`, local training scripts in `scripts/lq`, analysis summaries in `outputs/`, markdown docs in `docs/`

---

## File Map

- Baseline config and launch scripts:
  - `scripts/lq/run_train_x_mouth_v10_fsq_probe.sh`
  - `docs/lq_train_presets.md`
- Training and model code:
  - `scripts/lq/train.py`
  - `scripts/lq/model/network.py`
- Analysis and reporting:
  - `scripts/lq/analyze_checkpoint.py`
  - `docs/lq_progress.md`
  - `RESEARCH_PROGRESS.md`
- Planned new scripts / docs:
  - `scripts/lq/run_train_x_mouth_v11_*.sh`
  - `scripts/lq/run_train_y_mouth_v1_fsq_probe.sh`
  - `docs/lq_fsq_followup_notes.md`

## Global Rules For All Next Probes

- Keep the data roots fixed:
  - `data/win20-step20/IMR,data/win20-step20/TT`
- Keep the comparison setting fixed unless the task explicitly changes it:
  - `mode=x`
  - `region=mouth`
  - `batch_size=64`
  - `group_size=4`
  - `basis_size=119`
  - `action_basis_init_path=scripts/lq/init_basis/basis_x.npy`
- Every probe must produce:
  - `best.pt`
  - `analysis/summary.json`
  - one short markdown note with result and conclusion
- Promotion criterion for a new baseline:
  - code usage no worse than `v10`
  - validation reconstruction not worse by more than a small tolerated margin
  - change must answer a structural question, not only move one scalar metric

## Reference Baseline

- Baseline run:
  - `outputs/lq_x_mouth_v10_fsq_probe`
- Reference metrics:
  - `val_loss = 0.3238`
  - `val_recon = 0.3081`
  - `val_shared_recon = 0.3335`
  - `val_scaled_residual = 0.0329`
  - L1 `[361, 552]`
  - L2 `[335, 78, 500]`
  - L3 `[300, 33, 22, 36, 38, 484]`

### Task 1: Freeze `v10 FSQ` As The Official Baseline

**Files:**
- Modify: `docs/lq_train_presets.md`
- Modify: `docs/lq_progress.md`
- Modify: `RESEARCH_PROGRESS.md`
- Optional create: `docs/lq_fsq_followup_notes.md`

- [ ] **Step 1: Mark `v10` as the active baseline in docs**

Update docs so they say:
- current default quantizer baseline is `FSQ`
- old `LatentQuantize` runs are historical comparison runs
- all next probes should compare to `v10`

- [ ] **Step 2: Copy the exact baseline command into the notes**

Run to verify the command is still the canonical baseline:

```bash
sed -n '1,120p' scripts/lq/run_train_x_mouth_v10_fsq_probe.sh
```

Expected:
- the script uses `--quantizer_type=fsq`
- the script still points to the `x + mouth` basis init

- [ ] **Step 3: Record baseline success criteria**

Write the three baseline comparison axes into docs:
- reconstruction
- shared reconstruction
- code usage spread

- [ ] **Step 4: Commit**

```bash
git add docs/lq_train_presets.md docs/lq_progress.md RESEARCH_PROGRESS.md docs/lq_fsq_followup_notes.md
git commit -m "docs: promote fsq probe to lq baseline"
```

### Task 2: Add A Standard FSQ Baseline Analysis Template

**Files:**
- Modify: `scripts/lq/analyze_checkpoint.py`
- Create: `docs/lq_fsq_followup_notes.md`

- [ ] **Step 1: Define the standard comparison table**

The table must include:
- run name
- val loss
- val recon
- val shared recon
- val scaled residual
- per-level counts
- short conclusion

- [ ] **Step 2: Add any missing summary fields only if needed**

If the current `summary.json` lacks one comparison field needed repeatedly,
extend `scripts/lq/analyze_checkpoint.py` minimally.

Run:

```bash
python scripts/lq/analyze_checkpoint.py --checkpoint_path=outputs/lq_x_mouth_v10_fsq_probe/best.pt --output_dir=outputs/lq_x_mouth_v10_fsq_probe/analysis
```

Expected:
- analysis reruns without error
- `summary.json` still contains the baseline fields

- [ ] **Step 3: Create the running experiment log**

Start `docs/lq_fsq_followup_notes.md` with sections:
- baseline
- probe
- outcome
- decision

- [ ] **Step 4: Commit**

```bash
git add scripts/lq/analyze_checkpoint.py docs/lq_fsq_followup_notes.md
git commit -m "tools: standardize fsq follow-up reporting"
```

### Task 3: Run The First FSQ-Based Structural Probe

**Files:**
- Modify: `scripts/lq/model/network.py`
- Optional modify: `scripts/lq/train.py`
- Create: `scripts/lq/run_train_x_mouth_v11_*.sh`
- Modify: `docs/lq_train_presets.md`
- Modify: `docs/lq_fsq_followup_notes.md`

- [ ] **Step 1: Choose one structural question only**

The first FSQ-era structure probe should answer:
- can the shared path be strengthened relative to the private residual path

Recommended first probe:
- keep `FSQ`
- keep `x + mouth`
- reduce private-branch freedom before adding new supervision

Candidate directions:
- smaller `private_dim`
- shallower or weaker `private_decoder`
- stronger constraint on private residual magnitude

- [ ] **Step 2: Write the probe script**

Create one script only, for example:

```bash
scripts/lq/run_train_x_mouth_v11_private_branch_tighten.sh
```

Expected:
- only one structural change relative to `v10`
- output dir follows the same naming pattern

- [ ] **Step 3: Run a 1-epoch smoke test**

Run:

```bash
python scripts/lq/train.py --epochs=1 ...
```

Expected:
- memory check passes
- checkpoint saves
- analysis script can read it

- [ ] **Step 4: Run the full 15-epoch probe**

Run:

```bash
bash scripts/lq/run_train_x_mouth_v11_private_branch_tighten.sh
python scripts/lq/analyze_checkpoint.py --checkpoint_path=outputs/<new_run>/best.pt --output_dir=outputs/<new_run>/analysis
```

Expected:
- directly comparable summary to `v10`

- [ ] **Step 5: Decide promote / reject**

Promote only if:
- code usage stays broad or improves
- shared reconstruction improves or remains competitive
- residual dependence decreases in a meaningful way

- [ ] **Step 6: Commit**

```bash
git add scripts/lq/model/network.py scripts/lq/train.py scripts/lq/run_train_x_mouth_v11_private_branch_tighten.sh docs/lq_train_presets.md docs/lq_fsq_followup_notes.md
git commit -m "exp: probe fsq baseline with tighter private branch"
```

### Task 4: Reintroduce Auxiliary Supervision On Top Of FSQ

**Files:**
- Modify: `scripts/lq/train.py`
- Optional modify: `scripts/lq/model/network.py`
- Create: `scripts/lq/run_train_x_mouth_v12_*.sh`
- Modify: `docs/lq_fsq_followup_notes.md`

- [ ] **Step 1: Reintroduce one auxiliary signal at a time**

Do not re-enable everything together.

Recommended order:
1. continuous side supervision
2. discrete side supervision
3. dataset auxiliary heads

- [ ] **Step 2: Start with continuous side only**

Create a probe script that changes only:
- `side_cont_weight > 0`
- `side_disc_weight = 0`
- dataset aux still off

- [ ] **Step 3: Run smoke + full training**

Run:

```bash
bash scripts/lq/run_train_x_mouth_v12_side_cont_on_fsq.sh
python scripts/lq/analyze_checkpoint.py --checkpoint_path=outputs/<new_run>/best.pt --output_dir=outputs/<new_run>/analysis
```

Expected:
- no code usage collapse back to one dominant code

- [ ] **Step 4: Compare against `v10`**

Reject the auxiliary loss if:
- L2 or L3 usage collapses sharply again
- the gain is only on side loss but not on shared-motion behavior

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/train.py scripts/lq/model/network.py scripts/lq/run_train_x_mouth_v12_side_cont_on_fsq.sh docs/lq_fsq_followup_notes.md
git commit -m "exp: test side supervision on fsq baseline"
```

### Task 5: Validate The FSQ Trend On `mode=y`

**Files:**
- Create: `scripts/lq/run_train_y_mouth_v1_fsq_probe.sh`
- Modify: `docs/lq_train_presets.md`
- Modify: `docs/lq_fsq_followup_notes.md`

- [ ] **Step 1: Mirror the `v10` setup for `mode=y`**

Change only:
- `--mode=y`
- `--action_basis_init_path=scripts/lq/init_basis/basis_y.npy`

- [ ] **Step 2: Run a smoke test**

Run:

```bash
python scripts/lq/train.py --epochs=1 --mode=y --action_basis_init_path=scripts/lq/init_basis/basis_y.npy --quantizer_type=fsq ...
```

Expected:
- no shape mismatch
- no basis init mismatch

- [ ] **Step 3: Run the full `y + mouth` probe**

Run:

```bash
bash scripts/lq/run_train_y_mouth_v1_fsq_probe.sh
python scripts/lq/analyze_checkpoint.py --checkpoint_path=outputs/<new_run>/best.pt --output_dir=outputs/<new_run>/analysis
```

- [ ] **Step 4: Compare `x` vs `y`**

Decision question:
- is FSQ helping generally, or only in `x + mouth`

- [ ] **Step 5: Commit**

```bash
git add scripts/lq/run_train_y_mouth_v1_fsq_probe.sh docs/lq_train_presets.md docs/lq_fsq_followup_notes.md
git commit -m "exp: validate fsq baseline on y direction"
```

### Task 6: Revisit Dataset Semantics Only After The FSQ Baseline Stabilizes

**Files:**
- Modify: `scripts/lq/datasets.py`
- Modify: `docs/lq_dataset_refactor_checklist.md`
- Modify: `docs/lq_fsq_followup_notes.md`

- [ ] **Step 1: Do not mix dataset-semantic changes into structure probes**

Freeze dataset semantics while the first FSQ follow-up probes are running.

- [ ] **Step 2: After at least one successful FSQ follow-up probe, add the next dataset fix**

Recommended first semantic fix:
- `deleted_x` / `deleted_y` filtering

- [ ] **Step 3: Re-run the active baseline after dataset semantics change**

Run:

```bash
python scripts/lq/train.py --epochs=1 ...
bash scripts/lq/run_train_x_mouth_v10_fsq_probe.sh
python scripts/lq/analyze_checkpoint.py --checkpoint_path=outputs/lq_x_mouth_v10_fsq_probe/best.pt --output_dir=outputs/lq_x_mouth_v10_fsq_probe/analysis
```

Expected:
- if metrics move, the docs explicitly say the baseline changed because dataset semantics changed

- [ ] **Step 4: Commit**

```bash
git add scripts/lq/datasets.py docs/lq_dataset_refactor_checklist.md docs/lq_fsq_followup_notes.md
git commit -m "data: update lq dataset semantics after fsq baseline freeze"
```

## Recommended Execution Order

1. Task 1: freeze and document `v10`
2. Task 2: standardize result reporting
3. Task 3: run one FSQ-based structure probe
4. Task 4: reintroduce auxiliary supervision carefully
5. Task 5: check whether the FSQ effect transfers to `mode=y`
6. Task 6: only then revisit dataset semantics

## Stop Conditions

Stop and reassess if any of the following happens:

- a follow-up probe restores strong collapse similar to pre-FSQ runs
- shared reconstruction improves only by making the private branch stronger
- a dataset semantics change moves the baseline enough that old comparisons are no longer fair
- `mode=y` behaves qualitatively differently from `mode=x`

## Deliverables At The End Of This Phase

- one documented canonical FSQ baseline
- one FSQ-era structure probe result
- one FSQ-era supervision probe result
- one `mode=y` replication result
- updated docs explaining whether `FSQ` is a one-off fix or a robust new base
