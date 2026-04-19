#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python scripts/lq/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --epochs=50 \
  --batch_size=64 \
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
  --side_weight=0.0 \
  --side_cont_weight=0.0 \
  --side_disc_weight=0.0 \
  --private_residual_weight=0.05 \
  --private_residual_max_l1=0.5 \
  --shared_basis_soft_mixing=True \
  --shared_basis_anchor_bias=2.0 \
  --shared_basis_topk=2 \
  --quantizer_type=residual_fsq \
  --fsq_preserve_symmetry=True \
  --basis_orthogonalization=global_qr \
  --use_dataset_aux=False \
  --output_dir=outputs/lq_x_mouth_v19_residual_fsq_sparse_shared_tighter_private_probe_win20_e50
