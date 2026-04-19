# LQ Train Presets

Current canonical dataset roots for `scripts/lq`:

- `data/win20-step20/IMR,data/win20-step20/TT`

Refactor compatibility note:

- round-1 split the internal implementation across `scripts/lq/training/`,
  `scripts/lq/data/`, and `scripts/lq/model/`
- the training entrypoint and existing shell-script CLI shapes remain unchanged
- current presets in this document should still be launched exactly as before

## Conservative Preset

Current recommended first-pass training preset:

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `3.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_weight: `0.15`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v1.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v1
```

## Conservative Anti-Collapse Preset

This is the current next-step comparison preset after v2 showed that
`weight_decay + optimize_values=False` alone did not relieve collapse.

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.15`
- side_disc_weight: `0.0`
- weight_decay: `0.001`
- lq_commitment_loss_weight: `1.0`
- lq_quantization_loss_weight: `1.0`
- lq_optimize_values: `False`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v3_no_disc_side_stronger_lq.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v3_no_disc_side_stronger_lq
```

## Low-Residual Diagnostic Preset

This keeps the v3 official-LQ setup but constrains the private residual branch
so the shared action bases must explain more of the matrix directly.

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.15`
- side_disc_weight: `0.0`
- private_residual_weight: `0.05`
- weight_decay: `0.001`
- lq_commitment_loss_weight: `1.0`
- lq_quantization_loss_weight: `1.0`
- lq_optimize_values: `False`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v4_low_residual.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v4_low_residual
```

## Low-Residual Capped Preset

This extends v4 by capping the per-sample private residual mean absolute value,
so the model cannot recover the same escape route by inflating residual
amplitude while keeping a small residual weight.

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.15`
- side_disc_weight: `0.0`
- private_residual_weight: `0.05`
- private_residual_max_l1: `1.0`
- weight_decay: `0.001`
- lq_commitment_loss_weight: `1.0`
- lq_quantization_loss_weight: `1.0`
- lq_optimize_values: `False`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v5_low_residual_capped.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v5_low_residual_capped
```

## No-Side Probe Preset

This keeps the v5 capped-residual setup but disables side supervision entirely.
It is a structural probe for whether side labels are the main force collapsing
the shared discrete codes.

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.0`
- side_disc_weight: `0.0`
- private_residual_weight: `0.05`
- private_residual_max_l1: `1.0`
- weight_decay: `0.001`
- lq_commitment_loss_weight: `1.0`
- lq_quantization_loss_weight: `1.0`
- lq_optimize_values: `False`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v6_no_side_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v6_no_side_probe
```

## Shared-Bottleneck Probe Preset

This keeps the v6 no-side, capped-residual setup but shrinks the shared latent
from `32` to `8`. It tests whether the shared quantized path is collapsing
because its continuous pre-quantization representation is still too expressive.

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- hidden_dim: `32`
- shared_dim: `8`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.0`
- side_disc_weight: `0.0`
- private_residual_weight: `0.05`
- private_residual_max_l1: `1.0`
- weight_decay: `0.001`
- lq_commitment_loss_weight: `1.0`
- lq_quantization_loss_weight: `1.0`
- lq_optimize_values: `False`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v7_shared_bottleneck.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v7_shared_bottleneck
```

## Pool-2 Probe Preset

This keeps the v6 no-side, capped-residual baseline but replaces the final
`1x1` adaptive average pool with `2x2`, so the shared path sees a coarse spatial
layout instead of only one global mean per channel.

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- hidden_dim: `32`
- pool_size: `2`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.0`
- side_disc_weight: `0.0`
- private_residual_weight: `0.05`
- private_residual_max_l1: `1.0`
- weight_decay: `0.001`
- lq_commitment_loss_weight: `1.0`
- lq_quantization_loss_weight: `1.0`
- lq_optimize_values: `False`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v8_pool2_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v8_pool2_probe
```

## Soft-Basis Probe Preset

This keeps the v6 no-side, capped-residual baseline but upgrades shared
reconstruction from hard single-basis selection to per-level soft mixing over
all bases, with the discrete selected code used as an anchor bias.

- mode: `x`
- region: `mouth`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- hidden_dim: `32`
- pool_size: `1`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.0`
- side_disc_weight: `0.0`
- private_residual_weight: `0.05`
- private_residual_max_l1: `1.0`
- shared_basis_soft_mixing: `True`
- shared_basis_anchor_bias: `1.0`
- weight_decay: `0.001`
- lq_commitment_loss_weight: `1.0`
- lq_quantization_loss_weight: `1.0`
- lq_optimize_values: `False`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v9_soft_basis_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v9_soft_basis_probe
```

## FSQ Probe Preset

This is the current canonical FSQ baseline preset. It keeps the v6 no-side,
capped-residual structure and swaps `LatentQuantize` for official `FSQ`, which
matches the existing `levels=(2,3,6)` factorization more closely.

Current observed result on `win20-step20`:

- `val_loss = 0.3619`
- `val_recon = 0.3600`
- `val_shared_recon = 0.3620`
- L1 `[20, 60]`
- L2 `[20, 23, 37]`
- L3 `[18, 2, 3, 3, 25, 29]`

- mode: `x`
- region: `mouth`
- data_roots: `data/win20-step20/IMR,data/win20-step20/TT`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- hidden_dim: `32`
- pool_size: `1`
- quantizer_type: `fsq`
- fsq_preserve_symmetry: `True`
- use_dataset_aux: `False`
- recon_weight: `1.0`
- lq_weight: `10.0`
- orth_weight: `0.1`
- residual_weight: `0.02`
- side_cont_weight: `0.0`
- side_disc_weight: `0.0`
- private_residual_weight: `0.05`
- private_residual_max_l1: `1.0`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v10_fsq_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v10_fsq_probe_win20
```

## FSQ Private-Dim Probe

This is the first FSQ-era structure probe on top of the `win20-step20`
baseline. It changes only one variable relative to the canonical FSQ baseline:
`private_dim` is reduced from `32` to `8`.

Observed result:

- `val_loss = 0.3627`
- `val_recon = 0.3610`
- `val_shared_recon = 0.3625`
- `val_scaled_residual = 0.0026`
- L1 `[20, 60]`
- L2 `[0, 80, 0]`
- L3 `[19, 1, 4, 2, 54, 0]`

Decision:

- reject as the next baseline
- it reduced residual magnitude but made reconstruction slightly worse and
  collapsed L2 usage

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v11_private_dim8_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v11_private_dim8_probe_win20
```

## FSQ Private-Decoder Probe

This is the second FSQ-era structure probe on top of the `win20-step20`
baseline. It keeps `private_dim=32` and changes only the private decoder hidden
width from the baseline effective `64` down to `16`.

Observed result:

- `val_loss = 0.3636`
- `val_recon = 0.3615`
- `val_shared_recon = 0.3620`
- `val_scaled_residual = 0.0039`
- L1 `[20, 60]`
- L2 `[20, 22, 38]`
- L3 `[17, 3, 2, 3, 17, 38]`

Decision:

- reject as the next baseline
- less damaging than `private_dim=8`, but still not better than `v10`

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v12_private_decoder16_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v12_private_decoder16_probe_win20
```

## FSQ Side-Cont Probe

This is the first auxiliary-supervision probe on top of the `win20-step20`
FSQ baseline. It keeps the baseline structure fixed and enables only
continuous side supervision with `side_cont_weight=0.15`.

Observed result:

- `val_recon = 0.3623`
- `val_shared_recon = 0.3631`
- `val_scaled_residual = 0.0018`
- `val_side_cont = 1.0522`
- L1 `[8, 72]`
- L2 `[46, 34, 0]`
- L3 `[31, 28, 9, 11, 1, 0]`

Decision:

- reject as the next baseline
- it changes code allocation and lowers residual size, but does not improve
  shared-motion reconstruction

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v13_side_cont_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v13_side_cont_probe_win20
```

## FSQ Level-QR Probe

This probe keeps the canonical `win20-step20` FSQ baseline structure but
replaces the current per-basis normalization with strict QR orthogonalization
inside each level.

Observed result:

- `val_recon = 0.3574`
- `val_shared_recon = 0.3604`
- `val_scaled_residual = 0.0049`
- L1 `[20, 60]`
- L2 `[21, 59, 0]`
- L3 `[19, 2, 2, 2, 5, 50]`

Decision:

- mixed result, do not promote directly
- reconstruction improves, but code usage becomes more concentrated

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v14_level_qr_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v14_level_qr_probe_win20
```

Global-QR follow-up:

```bash
bash scripts/lq/run_train_x_mouth_v15_global_qr_probe.sh
```

Observed result:

- `val_loss = 0.3588`
- `val_recon = 0.3572`
- `val_shared_recon = 0.3597`
- `val_scaled_residual = 0.0041`
- L1 `[18, 62]`
- L2 `[19, 5, 56]`
- L3 `[58, 22, 0, 0, 0, 0]`

Decision:

- not promoted as the new baseline
- reconstruction improves further, but code usage becomes more concentrated than
  the level-wise QR probe

Expected output directory:

```bash
outputs/lq_x_mouth_v15_global_qr_probe_win20
```

## FSQ Global-QR + Basis-L1 Probe

This probe keeps the `v15 global_qr` structure and adds an L1 sparsity penalty
on the structured basis bank after QR projection.

- mode: `x`
- region: `mouth`
- data_roots: `data/win20-step20/IMR,data/win20-step20/TT`
- epochs: `15`
- batch_size: `64`
- group_size: `4`
- quantizer_type: `fsq`
- basis_orthogonalization: `global_qr`
- basis_l1_weight: `1.0`

Smoke-check observation:

- `basis_l1 = 0.00639`
- `orth ~= 0`
- forward + backward passed with `batch_size=64`, `group_size=4`

30-epoch observed result:

- `val_loss = 0.3478`
- `val_recon = 0.3310`
- `val_shared_recon = 0.3580`
- `val_scaled_residual = 0.0353`
- `val_basis_l1 = 0.00266`
- L1 `[56, 24]`
- L2 `[18, 7, 55]`
- L3 `[0, 0, 0, 0, 7, 73]`

Decision:

- not promoted as the interpretability baseline
- the sparsity prior works numerically, but the model compensates with a much
  larger private residual branch

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v16_global_qr_basis_l1_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v16_global_qr_basis_l1_probe_win20
```

## Residual-FSQ + Global-QR + Basis-L1 Probe

This probe replaces the single FSQ block with a residual FSQ stack while
keeping `global_qr` and basis sparsity enabled.

- mode: `x`
- region: `mouth`
- data_roots: `data/win20-step20/IMR,data/win20-step20/TT`
- epochs: `50`
- batch_size: `64`
- group_size: `4`
- quantizer_type: `residual_fsq`
- basis_orthogonalization: `global_qr`
- basis_l1_weight: `1.0`

Observed result:

- `val_loss = 0.3291`
- `val_recon = 0.3109`
- `val_shared_recon = 0.3423`
- `val_scaled_residual = 0.0393`
- `val_basis_l1 = 0.00239`
- L1 `[22, 58]`
- L2 `[50, 8, 22]`
- L3 `[55, 2, 1, 3, 13, 6]`

Decision:

- not promoted as the interpretability baseline
- residual FSQ improves higher-level code spread relative to `v16`, but shared
  reconstruction degrades substantially and the residual branch remains too
  dominant

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v17_residual_fsq_basis_l1_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v17_residual_fsq_basis_l1_probe_win20_e50
```

## Residual-FSQ + Sparse-Shared Probe

This probe keeps residual FSQ and sparse bases, then increases shared-path
capacity with anchor-guided sparse mixing and adds direct `shared_recon`
supervision.

- mode: `x`
- region: `mouth`
- data_roots: `data/win20-step20/IMR,data/win20-step20/TT`
- epochs: `50`
- batch_size: `64`
- group_size: `4`
- quantizer_type: `residual_fsq`
- shared_basis_soft_mixing: `True`
- shared_basis_anchor_bias: `2.0`
- shared_basis_topk: `2`
- shared_recon_weight: `1.0`
- basis_orthogonalization: `global_qr`
- basis_l1_weight: `1.0`

Observed result:

- `val_recon = 0.3150`
- `val_shared_recon = 0.3469`
- `val_scaled_residual = 0.0394`
- `val_basis_l1 = 0.00251`
- L1 `[22, 58]`
- L2 `[57, 1, 22]`
- L3 `[57, 1, 0, 2, 8, 12]`

Decision:

- promising direction, but not yet the interpretability baseline
- relative to `v17`, shared reconstruction improves materially, while private
  residual remains too large

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v18_residual_fsq_sparse_shared_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v18_residual_fsq_sparse_shared_probe_win20_e50
```

## Residual-FSQ + Sparse-Shared + Tighter-Private Probe

This probe keeps the `v18` shared-path design and tightens the private residual
cap to improve the interpretability tradeoff.

- mode: `x`
- region: `mouth`
- data_roots: `data/win20-step20/IMR,data/win20-step20/TT`
- epochs: `50`
- batch_size: `64`
- group_size: `4`
- quantizer_type: `residual_fsq`
- shared_basis_soft_mixing: `True`
- shared_basis_anchor_bias: `2.0`
- shared_basis_topk: `2`
- shared_recon_weight: `1.0`
- basis_orthogonalization: `global_qr`
- basis_l1_weight: `1.0`
- private_residual_max_l1: `0.5`

Observed result:

- `val_recon = 0.3275`
- `val_shared_recon = 0.3446`
- `val_scaled_residual = 0.0214`
- `val_basis_l1 = 0.00252`
- L1 `[22, 58]`
- L2 `[58, 0, 22]`
- L3 `[57, 1, 1, 1, 10, 10]`

Decision:

- promising interpretability tradeoff
- compared with `v18`, total reconstruction is worse, but private residual is
  much smaller while shared reconstruction remains materially better than `v17`
- keep this as the current safer interpretability baseline

Run it with:

```bash
bash scripts/lq/run_train_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe.sh
```

Expected output directory:

```bash
outputs/lq_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe_win20_e50
```

## Local Private-Cap Sweep Around `v19`

Two follow-up probes were run with the same shared-path structure as `v19`,
changing only `private_residual_max_l1`:

- `v20 cap=0.4`
  - `val_recon = 0.3310`
  - `val_shared_recon = 0.3448`
  - `val_scaled_residual = 0.0171`
  - L3 `[57, 1, 0, 2, 17, 3]`
- `v21 cap=0.6`
  - `val_recon = 0.3245`
  - `val_shared_recon = 0.3446`
  - `val_scaled_residual = 0.0251`
  - L3 `[57, 1, 0, 2, 11, 9]`

Interpretation:

- `cap=0.4` suppresses private residual best, but makes higher-level usage more
  concentrated than `v19`
- `cap=0.6` improves plain reconstruction, but mainly by letting the private
  branch grow again
- keep `v19 cap=0.5` as the default interpretability preset
- use `v20 cap=0.4` only when a stricter private-suppression ablation is needed

## Basis Init Mapping

Single-direction basis init files live in [`scripts/lq/init_basis`](/home/weizilin/generate_idea/scripts/lq/init_basis).

- `x + mouth`: `scripts/lq/init_basis/basis_x.npy`
- `y + mouth`: `scripts/lq/init_basis/basis_y.npy`
- `x + full`: `scripts/lq/init_basis/basis_x_full.npy`
- `y + full`: `scripts/lq/init_basis/basis_y_full.npy`

## Equivalent Direct Command

```bash
python scripts/lq/train.py \
  --epochs=15 \
  --batch_size=64 \
  --group_size=4 \
  --mode=x \
  --region=mouth \
  --basis_size=119 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x.npy \
  --recon_weight=1.0 \
  --lq_weight=3.0 \
  --orth_weight=0.1 \
  --residual_weight=0.02 \
  --side_weight=0.15 \
  --use_dataset_aux=False \
  --output_dir=outputs/lq_x_mouth_v1
```

## Minimal Variant Switches

- To run `y + mouth`, change:
  - `--mode=y`
  - `--action_basis_init_path=scripts/lq/init_basis/basis_y.npy`

- To run `x + full`, change:
  - `--region=full`
  - `--basis_size=341`
  - `--action_basis_init_path=scripts/lq/init_basis/basis_x_full.npy`

- To run `y + full`, change:
  - `--mode=y`
  - `--region=full`
  - `--basis_size=341`
  - `--action_basis_init_path=scripts/lq/init_basis/basis_y_full.npy`
