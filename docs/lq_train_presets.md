# LQ Train Presets

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
