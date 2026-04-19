# LQ FSQ Follow-Up Notes

Last updated: 2026-04-19

## Baseline

- canonical dataset roots: `data/win20-step20/IMR,data/win20-step20/TT`
- current target baseline run: `outputs/lq_x_mouth_v10_fsq_probe_win20`
- quantizer: `FSQ`
- mode: `x`
- region: `mouth`

## Historical Context

- earlier `v1` to `v10` runs were produced under the old
  `data/win10-step10/IMR,data/win10-step10/TT` setting
- those runs remain useful for structural comparison history
- they should not be treated as the final baseline for the new `win20-step20`
  dataset setting

## Probe Log

### Baseline Rerun: `v10 FSQ` on `win20-step20`

- status: completed
- command:

```bash
bash scripts/lq/run_train_x_mouth_v10_fsq_probe.sh
```

- expected outputs:
  - `outputs/lq_x_mouth_v10_fsq_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v10_fsq_probe_win20/analysis/summary.json`

- outcome:
  - `val_loss = 0.3619`
  - `val_recon = 0.3600`
  - `val_shared_recon = 0.3620`
  - `val_scaled_residual = 0.0034`
  - L1 `[20, 60]`
  - L2 `[20, 23, 37]`
  - L3 `[18, 2, 3, 3, 25, 29]`
  - total valid frames in validation analysis: `80`

- conclusion:
  - FSQ still gives non-collapsed code usage under `win20-step20`
  - compared with the older `win10-step10` round, reconstruction is weaker and
    the validation evidence is based on a much smaller number of valid frames
  - this run should still be treated as the active baseline for future probes
    because it matches the new canonical dataset setting

### Structure Probe: `v11 private_dim=8` on `win20-step20`

- status: completed
- command:

```bash
bash scripts/lq/run_train_x_mouth_v11_private_dim8_probe.sh
```

- expected outputs:
  - `outputs/lq_x_mouth_v11_private_dim8_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v11_private_dim8_probe_win20/analysis/summary.json`

- outcome:
  - `val_loss = 0.3627`
  - `val_recon = 0.3610`
  - `val_shared_recon = 0.3625`
  - `val_scaled_residual = 0.0026`
  - L1 `[20, 60]`
  - L2 `[0, 80, 0]`
  - L3 `[19, 1, 4, 2, 54, 0]`
  - total valid frames in validation analysis: `80`

- comparison against `v10`:
  - worse `val_loss`
  - worse `val_recon`
  - worse `val_shared_recon`
  - lower `scaled_residual`
  - noticeably worse code usage, especially L2 collapse from `[20, 23, 37]`
    to `[0, 80, 0]`

- conclusion:
  - shrinking `private_dim` from `32` to `8` is not a good first tightening
    strategy under the current `win20 + FSQ` setup
  - it reduces residual magnitude, but harms both reconstruction and code usage
  - this probe should be treated as rejected, not promoted

### Structure Probe: `v12 private_decoder_hidden_dim=16` on `win20-step20`

- status: completed
- command:

```bash
bash scripts/lq/run_train_x_mouth_v12_private_decoder16_probe.sh
```

- expected outputs:
  - `outputs/lq_x_mouth_v12_private_decoder16_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v12_private_decoder16_probe_win20/analysis/summary.json`

- outcome:
  - `val_loss = 0.3636`
  - `val_recon = 0.3615`
  - `val_shared_recon = 0.3620`
  - `val_scaled_residual = 0.0039`
  - L1 `[20, 60]`
  - L2 `[20, 22, 38]`
  - L3 `[17, 3, 2, 3, 17, 38]`
  - total valid frames in validation analysis: `80`

- comparison against `v10`:
  - worse `val_loss`
  - worse `val_recon`
  - nearly unchanged `val_shared_recon`
  - worse `scaled_residual`
  - code usage stays reasonably spread, but not clearly better than baseline

- conclusion:
  - narrowing private decoder width from the baseline effective `64` down to
    `16` does not improve the current `win20 + FSQ` setup
  - compared with `v11`, this direction is less destructive because it does not
    collapse L2, but it still fails to beat `v10`
  - this probe should also be treated as rejected

### Auxiliary Probe: `v13 side_cont_weight=0.15` on `win20-step20`

- status: completed
- command:

```bash
bash scripts/lq/run_train_x_mouth_v13_side_cont_probe.sh
```

- expected outputs:
  - `outputs/lq_x_mouth_v13_side_cont_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v13_side_cont_probe_win20/analysis/summary.json`

- important note:
  - `side_disc_weight=0.0`, so discrete side supervision is not optimized
  - however, once `side_labels` are passed in, the model still computes and
    reports `side_disc` metrics for inspection
  - because `side_cont` is included in total loss for this probe, total
    `val_loss` is not directly comparable to the baseline objective anymore

- outcome:
  - `val_recon = 0.3623`
  - `val_shared_recon = 0.3631`
  - `val_scaled_residual = 0.0018`
  - `val_side_cont = 1.0522`
  - L1 `[8, 72]`
  - L2 `[46, 34, 0]`
  - L3 `[31, 28, 9, 11, 1, 0]`
  - total valid frames in validation analysis: `80`

- comparison against `v10`:
  - worse reconstruction
  - worse shared reconstruction
  - lower scaled residual magnitude
  - code usage remains spread, but shifts substantially and does not yield a
    clearer shared-motion improvement

- conclusion:
  - reintroducing continuous side supervision at `0.15` does not improve the
    current `win20 + FSQ` baseline
  - it changes code allocation and lowers residual magnitude, but shared-motion
    behavior is not better and reconstruction becomes slightly worse
  - this probe should be treated as rejected for now

### Structure Probe: `v14 basis_orthogonalization=level_qr` on `win20-step20`

- status: completed
- command:

```bash
bash scripts/lq/run_train_x_mouth_v14_level_qr_probe.sh
```

- expected outputs:
  - `outputs/lq_x_mouth_v14_level_qr_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v14_level_qr_probe_win20/analysis/summary.json`

- important note:
  - this probe enforces strict orthonormality only within each level
  - the reported `orth` metric is still the global soft penalty across all 11
    bases, so it remains nonzero because cross-level similarity is still
    allowed and penalized

- outcome:
  - `val_recon = 0.3574`
  - `val_shared_recon = 0.3604`
  - `val_scaled_residual = 0.0049`
  - L1 `[20, 60]`
  - L2 `[21, 59, 0]`
  - L3 `[19, 2, 2, 2, 5, 50]`
  - total valid frames in validation analysis: `80`

- comparison against `v10`:
  - better reconstruction
  - better shared reconstruction
  - worse residual magnitude
  - noticeably more concentrated code usage, especially at L2 and L3

- conclusion:
  - level-wise QR is the first follow-up probe that improves shared-path
    reconstruction on top of the `win20 + FSQ` baseline
  - however, it does so while making code usage more concentrated
  - this should be treated as a mixed result, not a clean new baseline
  - the most likely next step is to keep QR optional and test coefficient-scale
    control on top of it, rather than promoting QR alone

### Structure Probe: `v15 basis_orthogonalization=global_qr` on `win20-step20`

- status: completed
- motivation:
  - `level_qr` only prevents basis similarity inside each level partition
  - if we want all 11 bases to be mutually distinct, the QR step must be
    applied once over the full basis bank

- implementation:
  - `scripts/lq/model/network.py` now supports
    `basis_orthogonalization=global_qr`
  - `global_qr` runs QR on the flattened full basis bank and then reshapes back
    to `(sum(levels), H, W)`
  - this means cross-level basis similarity is no longer permitted by the
    structured projection itself

- probe entry:

```bash
bash scripts/lq/run_train_x_mouth_v15_global_qr_probe.sh
```

- expected outputs:
  - `outputs/lq_x_mouth_v15_global_qr_probe_win20/best.pt`
  - `outputs/lq_x_mouth_v15_global_qr_probe_win20/analysis/summary.json`

- outcome:
  - `val_loss = 0.3588`
  - `val_recon = 0.3572`
  - `val_shared_recon = 0.3597`
  - `val_scaled_residual = 0.0041`
  - L1 `[18, 62]`
  - L2 `[19, 5, 56]`
  - L3 `[58, 22, 0, 0, 0, 0]`
  - total valid frames in validation analysis: `80`

- comparison against `v10`:
  - better total reconstruction
  - better shared reconstruction
  - somewhat larger residual magnitude
  - much more concentrated code usage, especially at L3

- comparison against `v14`:
  - slightly better total reconstruction
  - slightly better shared reconstruction
  - slightly lower residual magnitude
  - code usage is even more collapsed across higher levels

- conclusion:
  - full-bank QR successfully enforces the stronger structural prior that no
    two bases should be similar, even across levels
  - however, the current decoder/coeff pathway responds by using fewer discrete
    codes, not by spreading usage over the now-more-distinct basis bank
  - this should be treated as a structural confirmation, not a new baseline

## Decision

- active baseline: `outputs/lq_x_mouth_v10_fsq_probe_win20`
- rejected probe: `outputs/lq_x_mouth_v11_private_dim8_probe_win20`
- rejected probe: `outputs/lq_x_mouth_v12_private_decoder16_probe_win20`
- rejected probe: `outputs/lq_x_mouth_v13_side_cont_probe_win20`
- mixed probe: `outputs/lq_x_mouth_v14_level_qr_probe_win20`
- mixed probe: `outputs/lq_x_mouth_v15_global_qr_probe_win20`
- next recommended step:
  1. if optimizing the current `x + mouth` setup, stop tightening basis
     orthogonality and instead work on the shared coefficient / decoder side so
     the model can actually use more of the discrete bank
  2. otherwise, validate the FSQ trend on `mode=y`
  3. in parallel, revisit dataset/sample-efficiency strategy, since the
     validation set still has only `80` valid frames

### Structure Change: post-QR basis sparsity loss added

- status: completed as a 30-epoch probe
- motivation:
  - after `global_qr`, all bases are globally orthogonal, but they are still
    spatially dense
  - add an explicit sparsity prior on the structured basis bank itself

- implementation:
  - `scripts/lq/model/network.py` now exposes `basis_l1`
  - `basis_l1` is computed on the structured basis returned by
    `get_structured_basis()`, so the order is:
    1. enforce matrix structure
    2. apply QR projection when configured
    3. apply L1 sparsity loss on the resulting basis bank
  - the QR helper was also simplified to an EDTalk-style differentiable QR
    projection without the extra sign-correction step

- training hook:
  - `scripts/lq/train.py` now supports `basis_l1_weight`
  - probe entry:

```bash
bash scripts/lq/run_train_x_mouth_v16_global_qr_basis_l1_probe.sh
```

- smoke-check:
  - `basis_l1 = 0.00639`
  - `orth ~= 0`
  - forward + backward passed on `batch_size=64`, `group_size=4`

- 30-epoch outcome:
  - run: `outputs/lq_x_mouth_v16_global_qr_basis_l1_probe_win20_e30`
  - `val_loss = 0.3478`
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3580`
  - `val_scaled_residual = 0.0353`
  - `val_basis_l1 = 0.00266`
  - L1 `[56, 24]`
  - L2 `[18, 7, 55]`
  - L3 `[0, 0, 0, 0, 7, 73]`
  - total valid frames in validation analysis: `80`

- comparison against `v15`:
  - much better total reconstruction
  - slightly better shared reconstruction
  - dramatically larger private residual contribution
  - higher-level code usage becomes even more concentrated

- conclusion:
  - post-QR basis sparsity is effective numerically: basis L1 falls from the
    smoke-check `0.00639` to validation `0.00266`
  - however, the model pays for that sparsity by pushing much more load into
    the private residual branch
  - this is not a clean interpretability win, even though headline
    reconstruction improves

### Structure Probe: `v17 residual_fsq + global_qr + basis_l1` on `win20-step20`

- status: completed
- motivation:
  - replace the single FSQ block with a residual quantization stack
  - align stage 1 / 2 / 3 with the existing basis partitions `(2, 3, 6)`
  - keep `global_qr` and basis sparsity enabled

- implementation:
  - official `FSQ` blocks are stacked in residual form inside
    `scripts/lq/model/network.py`
  - stage 1 uses `levels=[2]`, stage 2 uses `levels=[3]`, stage 3 uses
    `levels=[6]`
  - each stage quantizes the remaining residual and feeds the matching basis
    branch

- probe entry:

```bash
bash scripts/lq/run_train_x_mouth_v17_residual_fsq_basis_l1_probe.sh
```

- outcome:
  - `val_loss = 0.3291`
  - `val_recon = 0.3109`
  - `val_shared_recon = 0.3423`
  - `val_scaled_residual = 0.0393`
  - `val_basis_l1 = 0.00239`
  - L1 `[22, 58]`
  - L2 `[50, 8, 22]`
  - L3 `[55, 2, 1, 3, 13, 6]`
  - total valid frames in validation analysis: `80`

- comparison against `v16`:
  - much better total reconstruction
  - much worse shared reconstruction
  - slightly larger private residual contribution
  - noticeably healthier higher-level code usage, especially at L3

- conclusion:
  - residual FSQ improves discrete code utilization relative to the collapsed
    `v16` sparse-basis run
  - however, it still does not solve the main structural issue: the model keeps
    improving total reconstruction by leaning on the private residual branch,
    while shared reconstruction gets worse
  - treat this as a useful structural probe, not as the new interpretability
    baseline

### Structure Probe: `v18 residual_fsq + sparse_shared_mixing + shared_recon_loss` on `win20-step20`

- status: completed
- motivation:
  - keep residual FSQ
  - increase shared-path expressivity with anchor-guided sparse mixing
  - add direct optimization pressure on `shared_recon`

- implementation:
  - `shared_basis_soft_mixing=True`
  - `shared_basis_anchor_bias=2.0`
  - `shared_basis_topk=2`
  - `shared_recon_weight=1.0`

- probe entry:

```bash
bash scripts/lq/run_train_x_mouth_v18_residual_fsq_sparse_shared_probe.sh
```

- outcome:
  - `val_recon = 0.3150`
  - `val_shared_recon = 0.3469`
  - `val_scaled_residual = 0.0394`
  - `val_basis_l1 = 0.00251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 8, 12]`
  - total valid frames in validation analysis: `80`

- important note:
  - `val_loss` is not directly comparable to older probes because this run adds
    `shared_recon_weight=1.0` into the total objective

- comparison against `v17`:
  - slightly worse total reconstruction
  - clearly better shared reconstruction
  - private residual magnitude remains high, roughly unchanged
  - code usage stays multi-code at L3, but is somewhat less spread than `v17`

- conclusion:
  - this validates the structural direction: increasing shared-path capacity
    and directly supervising `shared_recon` does pull the model away from the
    worst shared-path collapse
  - however, it is not yet enough to reduce reliance on the private residual
    branch
  - the next step should keep this shared-path design and explicitly tighten
    the private branch again, now that the shared branch has more capacity

### Structure Probe: `v19 v18 + tighter private residual cap` on `win20-step20`

- status: completed
- motivation:
  - keep the improved shared-path design from `v18`
  - tighten the private residual branch and test whether the model can retain
    better shared reconstruction with smaller private correction

- implementation:
  - same as `v18`, except `private_residual_max_l1=0.5`

- probe entry:

```bash
bash scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh
```

- outcome:
  - `val_recon = 0.3275`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0214`
  - `val_basis_l1 = 0.00252`
  - L1 `[22, 58]`
  - L2 `[58, 0, 22]`
  - L3 `[57, 1, 1, 1, 10, 10]`
  - total valid frames in validation analysis: `80`

- important note:
  - `val_loss` is still not directly comparable to older probes because this
    run keeps `shared_recon_weight=1.0`

- comparison against `v18`:
  - worse total reconstruction
  - slightly worse shared reconstruction
  - much lower private residual contribution
  - similar code usage pattern with slightly stronger concentration at L2

- comparison against `v17`:
  - worse total reconstruction
  - clearly better shared reconstruction
  - much lower private residual contribution

- conclusion:
  - this is the first probe in the current stage that meaningfully improves the
    interpretability tradeoff
  - the shared path remains much stronger than `v17`, while the private branch
    is no longer dominating at the same magnitude as `v17` / `v18`
  - the next step should search locally around this regime, rather than going
    back to unconstrained private residuals

### Structure Probe: `v20 v19 + tighter cap=0.4` on `win20-step20`

- status: completed
- motivation:
  - test whether the `v19` regime still improves if the private residual cap is
    tightened a bit further
  - keep the shared-path design fixed so the comparison isolates cap strength

- implementation:
  - same as `v19`, except `private_residual_max_l1=0.4`

- probe entry:

```bash
bash scripts/lq/run_train_x_mouth_v20_residual_fsq_sparse_shared_privatecap04_probe.sh
```

- outcome:
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3448`
  - `val_scaled_residual = 0.0171`
  - `val_basis_l1 = 0.00252`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 17, 3]`
  - total valid frames in validation analysis: `80`

- comparison against `v19`:
  - worse total reconstruction
  - slightly worse shared reconstruction
  - lower private residual contribution
  - higher-level usage shifts more strongly toward one late-stage code

- conclusion:
  - `cap=0.4` is a viable stricter-private variant, but not a clean upgrade
  - it improves the private-suppression metric, yet gives up a bit of shared
    quality and higher-level code spread

### Structure Probe: `v21 v19 + looser cap=0.6` on `win20-step20`

- status: completed
- motivation:
  - test the opposite side of the local tradeoff window around `v19`
  - check whether a slightly looser private cap can improve shared behavior
    rather than only plain reconstruction

- implementation:
  - same as `v19`, except `private_residual_max_l1=0.6`

- probe entry:

```bash
bash scripts/lq/run_train_x_mouth_v21_residual_fsq_sparse_shared_privatecap06_probe.sh
```

- outcome:
  - `val_recon = 0.3245`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0251`
  - `val_basis_l1 = 0.00251`
  - L1 `[22, 58]`
  - L2 `[57, 1, 22]`
  - L3 `[57, 1, 0, 2, 11, 9]`
  - total valid frames in validation analysis: `80`

- comparison against `v19`:
  - better total reconstruction
  - essentially unchanged shared reconstruction
  - clearly larger private residual contribution
  - L3 spread is acceptable, but the gain is bought mostly through the private
    branch rather than a stronger shared path

- conclusion:
  - `cap=0.6` is too loose for the current interpretability objective
  - it should not replace `v19` as the current baseline

### Local Sweep Conclusion: `private_residual_max_l1 in {0.4, 0.5, 0.6}`

- `v19 (cap=0.5)` remains the safest interpretability baseline
- `v20 (cap=0.4)` is the stricter-private alternative when lower
  `scaled_residual` is the first priority
- `v21 (cap=0.6)` confirms that relaxing the cap mostly restores private
  correction, not shared explanatory power
