# LQ Prototype Progress

Last updated: 2026-04-20 (round-1 side semantic bank smoke)

## Scope

This document records the recent implementation and experiment progress for
`scripts/lq`, focused on the current single-direction setup:

- mode: `x`
- region: `mouth`
- data: `data/win20-step20/IMR,data/win20-step20/TT`
- input: `B x T x 1 x H x W`
- current working target: shared discrete motion basis learning with per-frame
  reconstruction loss

It is meant as the working-note companion to
[`RESEARCH_PROGRESS.md`](/home/weizilin/generate_idea/RESEARCH_PROGRESS.md).

Round-1 refactor note:

- the internals of `scripts/lq` are now split into `training/`, `data/`, and
  `model/` subpackages
- public entrypoints remain unchanged:
  - `python scripts/lq/train.py ...`
  - `python scripts/lq/analyze_checkpoint.py ...`
- current dataset sample fields, checkpoint fields, and run-script semantics are
  intentionally preserved

Important note:

- the `v1` to `v10` metrics recorded below were produced during the earlier
  `win10-step10` round
- the canonical dataset roots have now been switched to `win20-step20`
- the FSQ baseline has now been rerun under the new dataset setting and should
  be treated as the authoritative comparison point for future work

Current canonical baseline:

- run: `outputs/lq_x_mouth_v10_fsq_probe_win20`
- `val_loss = 0.3619`
- `val_recon = 0.3600`
- `val_shared_recon = 0.3620`
- `val_scaled_residual = 0.0034`
- code usage:
  - L1 `[20, 60]`
  - L2 `[20, 23, 37]`
  - L3 `[18, 2, 3, 3, 25, 29]`

Important caution:

- the `win20-step20` validation split currently contains only `80` valid frames
- the new baseline still shows broad FSQ code usage, but this evidence is based
  on a much smaller validation set than the old `win10-step10` round

First FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v11_private_dim8_probe_win20`
- change: `private_dim 32 -> 8`
- result:
  - `val_loss = 0.3627`
  - `val_recon = 0.3610`
  - `val_shared_recon = 0.3625`
  - `val_scaled_residual = 0.0026`
  - L2 collapsed to `[0, 80, 0]`
- decision:
  - reject this direction as the next baseline
  - shrinking private latent width directly is too destructive under the
    current `win20 + FSQ` setting

Second FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v12_private_decoder16_probe_win20`
- change: private decoder hidden width reduced from the baseline effective `64`
  to `16`
- result:
  - `val_loss = 0.3636`
  - `val_recon = 0.3615`
  - `val_shared_recon = 0.3620`
  - `val_scaled_residual = 0.0039`
  - L2 `[20, 22, 38]`
  - L3 `[17, 3, 2, 3, 17, 38]`
- decision:
  - reject this direction as well
  - it avoids the severe L2 collapse seen in `v11`, but still does not improve
    over the `v10` baseline

First FSQ-era auxiliary probe on `win20`:

- run: `outputs/lq_x_mouth_v13_side_cont_probe_win20`
- change: enable `side_cont_weight=0.15` while keeping `side_disc_weight=0.0`
- result:
  - `val_recon = 0.3623`
  - `val_shared_recon = 0.3631`
  - `val_scaled_residual = 0.0018`
  - `val_side_cont = 1.0522`
  - L2 `[46, 34, 0]`
  - L3 `[31, 28, 9, 11, 1, 0]`
- decision:
  - reject this direction for now
  - continuous side supervision does not improve shared-motion reconstruction on
    the current `win20 + FSQ` baseline

Third FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v14_level_qr_probe_win20`
- change: replace per-basis normalization with strict level-wise QR
- result:
  - `val_recon = 0.3574`
  - `val_shared_recon = 0.3604`
  - `val_scaled_residual = 0.0049`
  - L2 `[21, 59, 0]`
  - L3 `[19, 2, 2, 2, 5, 50]`
- decision:
  - mixed result, do not promote directly
  - QR improves reconstruction and shared reconstruction, but code usage becomes
    more concentrated

Current follow-up adjustment:

- `level_qr` only removes similarity inside each level
- a new `global_qr` mode has been added in `scripts/lq/model/network.py`
- `global_qr` performs one QR over all 11 bases together, so cross-level basis
  similarity is no longer allowed
- probe entry: `scripts/lq/run_train_x_mouth_v15_global_qr_probe.sh`

Fourth FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v15_global_qr_probe_win20`
- change: replace level-wise QR with one global QR over all 11 bases
- result:
  - `val_loss = 0.3588`
  - `val_recon = 0.3572`
  - `val_shared_recon = 0.3597`
  - `val_scaled_residual = 0.0041`
  - L1 `[18, 62]`
  - L2 `[19, 5, 56]`
  - L3 `[58, 22, 0, 0, 0, 0]`
- decision:
  - treat as a useful structural confirmation, not a new baseline
  - full-bank QR improves total reconstruction again, but discrete usage becomes
    even more concentrated than `v14`

Fifth FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v16_global_qr_basis_l1_probe_win20_e30`
- change: keep full-bank QR and add basis L1 sparsity after QR
- result:
  - `val_loss = 0.3478`
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3580`
  - `val_scaled_residual = 0.0353`
  - `val_basis_l1 = 0.00266`
  - L1 `[56, 24]`
  - L2 `[18, 7, 55]`
  - L3 `[0, 0, 0, 0, 7, 73]`
- decision:
  - do not promote as the interpretability baseline
  - sparsity improves total reconstruction, but it does so mainly by allowing
    the private residual branch to grow much larger

Sixth FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v17_residual_fsq_basis_l1_probe_win20_e50`
- change: replace single FSQ with residual stage-wise FSQ while keeping
  `global_qr + basis_l1`
- result:
  - `val_loss = 0.3291`
  - `val_recon = 0.3109`
  - `val_shared_recon = 0.3423`
  - `val_scaled_residual = 0.0393`
  - `val_basis_l1 = 0.00239`
  - L1 `[22, 58]`
  - L2 `[50, 8, 22]`
  - L3 `[55, 2, 1, 3, 13, 6]`
- decision:
  - do not promote as the interpretability baseline
  - residual FSQ spreads higher-level code usage better than `v16`, but total
    improvement still comes with much worse shared reconstruction and a large
    private residual branch

Seventh FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v18_residual_fsq_sparse_shared_probe_win20_e50`
- change: keep residual FSQ, add anchor-guided sparse shared mixing, and add
  direct `shared_recon` supervision
- result:
  - `val_recon = 0.3150`
  - `val_shared_recon = 0.3469`
  - `val_scaled_residual = 0.0394`
  - `val_basis_l1 = 0.00251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 8, 12]`
- decision:
  - useful positive direction, but not yet the interpretability baseline
  - compared with `v17`, shared reconstruction improves clearly, but private
    residual remains too large

Eighth FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe_win20_e50`
- change: keep `v18` shared design and tighten `private_residual_max_l1` from
  `1.0` to `0.5`
- result:
  - `val_recon = 0.3275`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0214`
  - `val_basis_l1 = 0.00252`
  - L1 `[22, 58]`
  - L2 `[58, 0, 22]`
  - L3 `[57, 1, 1, 1, 10, 10]`
- decision:
  - promising interpretability tradeoff
  - compared with `v18`, private residual drops substantially while shared
    reconstruction remains much better than `v17`

Ninth FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v20_residual_fsq_sparse_shared_privatecap04_probe_win20_e50`
- change: keep `v19` structure and tighten `private_residual_max_l1` further
  from `0.5` to `0.4`
- result:
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3448`
  - `val_scaled_residual = 0.0171`
  - `val_basis_l1 = 0.00252`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 17, 3]`
- decision:
  - useful stricter-private variant, but not a clean replacement for `v19`
  - it pushes `scaled_residual` lower than `v19`, but `shared_recon` becomes
    slightly worse and L3 usage becomes more concentrated

Tenth FSQ-era structure probe on `win20`:

- run: `outputs/lq_x_mouth_v21_residual_fsq_sparse_shared_privatecap06_probe_win20_e50`
- change: keep `v19` structure and loosen `private_residual_max_l1` from `0.5`
  to `0.6`
- result:
  - `val_recon = 0.3245`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0251`
  - `val_basis_l1 = 0.00251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 11, 9]`
- decision:
  - reject as the next interpretability baseline
  - it improves plain reconstruction, but mainly by allowing the private branch
    to grow again; shared reconstruction is not meaningfully better than
    `v19/v20`

Current local sweep readout around `v19`:

- `cap=0.4` lowers `scaled_residual` best, but concentrates higher-level usage
  more strongly
- `cap=0.5` remains the safest interpretability baseline because it keeps a
  better balance between private suppression and L3 spread
- `cap=0.6` is too loose for the current objective and mostly restores private
  correction capacity

First round-1 side semantic bank smoke on `win20`:

- run: `outputs/lq_x_mouth_v22_side_semantic_bank_probe_smoke`
- preset change:
  - keep the `v19` backbone intact:
    - `action_basis_init_path=scripts/lq/init_basis/basis_x.npy`
    - `shared_recon_weight=1.0`
    - `quantizer_type=residual_fsq`
    - `basis_orthogonalization=global_qr`
    - `private_residual_max_l1=0.5`
  - enable the round-1 side semantic bank only:
    - `side_semantic_enabled=True`
    - `side_basis_count=2`
    - `side_loss_weight=0.3`
- smoke result:
  - `val_loss = 1.0672`
  - `val_recon = 0.3636`
  - `val_shared_recon = 0.3636`
  - `val_scaled_residual = 0.00233`
  - `val_side_group = 1.1104`
- analysis result:
  - output files written:
    - `analysis/summary.json`
    - `analysis/side_basis_bank_heatmap.png`
    - `analysis/group_level_representations.npz`
  - `side_basis_shape = [2, 119, 119]`
  - `mean_side_path_usage = 0.500`
  - `mean_free_path_usage = 0.273`
  - `mean_side_recon_l1 = 0.00162`
  - `mean_free_recon_l1 = 0.000262`
  - `side_from_side_rep_acc = 0.364`
  - `side_from_free_rep_acc = 0.364`
  - `dataset_from_side_rep_acc = 0.818`
  - `dataset_from_free_rep_acc = 0.818`
- decision:
  - accept as a successful round-1 preset and analysis smoke
  - do not interpret it as semantic separation success yet
  - `B_side` is not idle, but after one epoch it does not beat the free path on
    the side probe, and both shared branches still retain obvious dataset signal

## Current Implementation Status

### 1. Dataset / Input Pipeline

Current status:

- `scripts/lq/datasets.py` now supports grouped sequence loading for training.
- The training input is aligned to `batch x win x matrix_size x matrix_size`.
- Current experiments use `group_size=4`.
- A batch-memory smoke check has been added before training starts.

Validated:

- With `batch_size=64`, `group_size=4`, `mode=x`, `region=mouth`,
  one input batch has shape `(64, 4, 1, 119, 119)`.
- The input tensor itself is about `13.83 MiB`.
- Forward + backward smoke test passed without OOM.

Known remaining dataset issues:

- `deleted_x` / `deleted_y` semantics are not fully integrated yet.
- Dataset metadata is still minimal for downstream analysis.
- Sampling is not yet patient-balanced or dataset-balanced.

Related checklist:

- [`docs/lq_dataset_refactor_checklist.md`](/home/weizilin/generate_idea/docs/lq_dataset_refactor_checklist.md)

### 2. Training Pipeline

Implemented:

- `scripts/lq/train.py` now accepts sequence input directly.
- Loss is computed per frame, then reduced with valid/padding masks.
- `basis_init` is required by default for the current training stage.
- Batch-memory validation is run before full training.
- round-1 refactor has split training internals into `scripts/lq/training/`
  while preserving the current CLI signature and metric keys

Tracked metrics now include:

- `loss`
- `recon`
- `shared_recon`
- `lq`
- `orth`
- `residual`
- `scaled_residual`
- optional side / dataset losses

### 3. Model Structure

Current model file:

- [`scripts/lq/model/network.py`](/home/weizilin/generate_idea/scripts/lq/model/network.py)
- `network.py` is now a thin compatibility layer over the split model modules in
  `scripts/lq/model/`

Implemented structural changes during this round:

- sequence input flatten / restore logic
- configurable `pool_size`
- configurable `shared_dim`
- capped private residual with `private_residual_max_l1`
- optional soft basis mixing with anchor bias
- official quantizer switch:
  - `quantizer_type="latent_quantize"`
  - `quantizer_type="fsq"`

Current working interpretation:

- shared path: discrete motion code + action basis reconstruction
- private path: residual correction branch

### 4. Analysis / Tooling

Implemented:

- `scripts/lq/analyze_checkpoint.py` can now load newer checkpoints with
  `pool_size`, `shared_dim`, quantizer config, and residual cap config.
- basis bank heatmap and code-usage summary are generated automatically.
- Pillow resampling compatibility warning has been removed.
- round-1 refactor keeps analysis CLI unchanged while switching imports to the
  split `data/` and `model/` packages

## Experiment Timeline

All experiments below are under the same basic setting unless noted:

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`

### v1. Conservative baseline

Output:

- `outputs/lq_x_mouth_v1`

Result:

- `val_loss = 0.6309`
- code usage already collapsed:
  - L1 `[0, 913]`
  - L2 `[0, 913, 0]`
  - L3 `[0, 0, 0, 0, 788, 125]`

Conclusion:

- Baseline trains, but collapse appears immediately.

### v2. Official LatentQuantize anti-collapse probe

Change:

- switch to official `LatentQuantize`
- use anti-collapse-oriented settings such as stronger weight decay and
  `optimize_values=False`

Output:

- `outputs/lq_x_mouth_v2_official_lq_anticollapse`

Result:

- `val_loss = 0.6308`
- code usage remains collapsed:
  - L1 `[0, 913]`
  - L2 `[0, 913, 0]`
  - L3 `[0, 0, 0, 0, 803, 110]`

Conclusion:

- Official LQ settings alone did not solve collapse.

### v3. No discrete side + stronger LQ

Change:

- remove discrete side supervision
- increase LQ pressure

Output:

- `outputs/lq_x_mouth_v3_no_disc_side_stronger_lq`

Result:

- `val_loss = 0.4414`
- L3 usage still nearly single-code: `[0, 0, 0, 0, 903, 10]`

Conclusion:

- Better validation loss, but collapse still severe.

### v4. Low-residual probe

Change:

- reduce private residual contribution weight

Output:

- `outputs/lq_x_mouth_v4_low_residual`

Result:

- `val_loss = 0.4974`
- L3 `[0, 0, 0, 0, 898, 15]`

Conclusion:

- Lower residual weight did not fix collapse.
- The model could still exploit residual amplitude.

### v5. Low-residual capped probe

Change:

- cap private residual mean absolute magnitude with
  `private_residual_max_l1=1.0`

Output:

- `outputs/lq_x_mouth_v5_low_residual_capped`

Result:

- `val_loss = 0.5112`
- L3 `[0, 0, 0, 0, 896, 17]`

Conclusion:

- Residual bypass was constrained, but collapse still remained.

### v6. No-side probe

Change:

- disable side supervision entirely

Output:

- `outputs/lq_x_mouth_v6_no_side_probe`

Result:

- `val_loss = 0.3462`
- `val_recon = 0.3255`
- `val_shared_recon = 0.3541`
- `val_scaled_residual = 0.0392`
- code usage:
  - L1 `[0, 913]`
  - L2 `[0, 913, 0]`
  - L3 `[0, 0, 0, 0, 891, 22]`

Conclusion:

- Side supervision is not the main reason for collapse.
- This became the main structural comparison baseline.

### v7. Shared bottleneck probe

Change:

- shrink `shared_dim` from `32` to `8`

Output:

- `outputs/lq_x_mouth_v7_shared_bottleneck`

Result:

- `val_loss = 0.3446`
- L3 fully collapsed to one code: `[0, 0, 0, 0, 0, 913]`

Conclusion:

- Narrowing the shared latent alone made collapse worse.

### v8. Pool-2 probe

Change:

- replace `AdaptiveAvgPool2d((1, 1))` with `AdaptiveAvgPool2d((2, 2))`

Output:

- `outputs/lq_x_mouth_v8_pool2_probe`

Result:

- `val_loss = 0.3588`
- L2 slightly spread: `[17, 896, 0]`
- L3 still collapsed: `[0, 0, 0, 0, 913, 0]`

Conclusion:

- Pooling scale affected early behavior, but did not fundamentally solve
  collapse.

### v9. Soft-basis probe

Change:

- replace hard per-level basis pick with soft mixing over each level, using the
  discrete selected code as anchor bias

Output:

- `outputs/lq_x_mouth_v9_soft_basis_probe`

Result:

- `val_loss = 0.3458`
- L3 `[0, 0, 0, 0, 904, 9]`

Conclusion:

- Relaxing basis selection alone was not enough.

### v10. Official FSQ replacement

Change:

- replace `LatentQuantize` with official `FSQ`
- keep the rest of the v6-style structural constraints comparable

Output:

- `outputs/lq_x_mouth_v10_fsq_probe`

Result:

- `val_loss = 0.3238`
- `val_recon = 0.3081`
- `val_shared_recon = 0.3335`
- `val_scaled_residual = 0.0329`
- code usage:
  - L1 `[361, 552]`
  - L2 `[335, 78, 500]`
  - L3 `[300, 33, 22, 36, 38, 484]`

Conclusion:

- This is the first experiment that clearly improves code usage instead of only
  changing reconstruction metrics.
- FSQ not only runs stably, but materially reduces collapse under the current
  architecture.
- In the older `win10-step10` round, `v10` became the new baseline.
- In the current `win20-step20` round, the rerun still shows broad code usage
  and remains the active baseline, but its validation set is much smaller.

## Current Main Findings

### What did not fix collapse

- side supervision changes
- stronger official `LatentQuantize` settings
- lowering private residual weight alone
- residual cap alone
- shrinking shared latent dimension
- increasing pooling size alone
- soft basis mixing alone

### What did work

- switching the shared quantizer to official `FSQ`

At the current stage, the evidence supports this reading:

- the earlier collapse was not just a loss-weight problem
- it was also not explained by one single structural detail such as
  `1x1` pooling or side supervision
- the choice of quantizer is a major factor in whether the shared discrete path
  can use its codebook meaningfully

## Current Risks And Open Questions

1. `v10` improved code usage a lot, but the shared path is still weaker than the
   full reconstruction path.
   - `val_shared_recon = 0.3335`
   - `val_recon = 0.3081`

2. Improvement still coexists with a nontrivial private residual branch.
   - The structure is healthier than before, but disentanglement is not yet
     proven.

3. FSQ currently has no explicit LQ penalty term in our metric layout.
   - In the current implementation, `lq_loss_per_sample` is zero for the FSQ
     path.
   - This is acceptable for the current comparison, but should stay explicit in
     later analysis.

4. Dataset semantics are still not fully finalized.
   - especially `deleted_x` / `deleted_y`

## Recommended Next Step

Use `v10 FSQ` on `win20-step20` as the active baseline, then continue
structural analysis on top of it instead of going back to `LatentQuantize`.

The next reasonable questions are:

1. whether the shared path should be strengthened further relative to the
   private residual path
2. whether side / dataset auxiliary heads should be reintroduced on top of FSQ
3. whether the same trend holds for `mode=y` and later `full` region settings
