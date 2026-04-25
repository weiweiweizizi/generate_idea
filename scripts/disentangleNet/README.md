# disentangleNet v31

`scripts/disentangleNet` is a self-contained freeze of the accepted `v31` training stack from `scripts/lq`, with the post-hoc probe analysis path preserved.

The training entrypoint is now intentionally narrowed to `v31`: only runtime knobs such as data roots, epochs, batch size, checkpoint output, and basis-init paths stay configurable. Experimental loss switches that do not affect `v31` are no longer exposed at the CLI.

## Included

- `train.py`: frozen training entrypoint for the final `v31` setup
- `data/`, `model/`, `training/`: runtime closure required by `v31`
- `init_basis/`: the two basis initializers used by the `v31` run script
- `analysis/`: retained probe and interpretability entrypoints

## Deliberately removed

- old compatibility shims such as `scripts/lq/model/network.py`
- old extraction shims such as `scripts/lq/datasets.py`
- unrelated experiment launchers and analysis scripts outside the `v31` path
- `scripts/lq` fallback imports inside the extracted package

## Train v31

```bash
bash scripts/disentangleNet/run_train_x_mouth_v31_joint_qr_levels26_side3_sparse_side_probe.sh
```

## Analyze a checkpoint

```bash
python scripts/disentangleNet/analysis/analyze_checkpoint.py \
  --checkpoint_path=outputs/disentangleNet/v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/best.pt

python scripts/disentangleNet/analysis/analyze_side_interpretability.py \
  --checkpoint_path=outputs/disentangleNet/v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/best.pt

python scripts/disentangleNet/analysis/analyze_kfold_report.py \
  --checkpoint_path=outputs/disentangleNet/v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50/best.pt
```

## v31 preserved settings

- `mode=x`
- `region=mouth`
- `levels=2,6`
- `quantizer_type=residual_fsq`
- `basis_orthogonalization=joint_global_qr`
- `side_semantic_enabled=True`
- `side_basis_count=3`
- `side_pooling=fixed_region2_contrast`
- `early_branch_factorization=True`
