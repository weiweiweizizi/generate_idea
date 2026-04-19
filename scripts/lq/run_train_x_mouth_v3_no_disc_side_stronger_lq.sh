#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python scripts/lq/train.py \
  --epochs=15 \
  --batch_size=64 \
  --group_size=4 \
  --mode=x \
  --region=mouth \
  --basis_size=119 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x.npy \
  --recon_weight=1.0 \
  --lq_weight=10.0 \
  --orth_weight=0.1 \
  --residual_weight=0.02 \
  --side_weight=0.15 \
  --side_cont_weight=0.15 \
  --side_disc_weight=0.0 \
  --weight_decay=0.001 \
  --lq_commitment_loss_weight=1.0 \
  --lq_quantization_loss_weight=1.0 \
  --lq_optimize_values=False \
  --use_dataset_aux=False \
  --output_dir=outputs/lq_x_mouth_v3_no_disc_side_stronger_lq
