#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

set +u
source /home/weizilin/anaconda3/etc/profile.d/conda.sh
conda activate dl
set -u

python scripts/lq/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --group_size=4 \
  --mode=x \
  --region=mouth \
  --basis_size=119 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x.npy \
  --hidden_dim=32 \
  --private_dim=32 \
  --recon_weight=1.0 \
  --shared_recon_weight=1.0 \
  --lq_weight=10.0 \
  --orth_weight=0.1 \
  --basis_l1_weight=1.0 \
  --residual_weight=0.02 \
  --private_residual_weight=0.05 \
  --private_residual_max_l1=0.5 \
  --shared_basis_soft_mixing=True \
  --shared_basis_anchor_bias=2.0 \
  --shared_basis_topk=2 \
  --quantizer_type=residual_fsq \
  --fsq_preserve_symmetry=True \
  --basis_orthogonalization=global_qr \
  --epochs=50 \
  --batch_size=64 \
  --side_semantic_enabled=True \
  --side_basis_count=2 \
  --side_z_dim=8 \
  --side_loss_weight=0.3 \
  --use_dataset_aux=False \
  --early_branch_factorization=True \
  --free_pool_size=2 \
  --side_pool_size=2 \
  --private_pool_size=1 \
  --output_dir=outputs/lq_x_mouth_v27_side_basis_rep_tight_probe_win20_e50
