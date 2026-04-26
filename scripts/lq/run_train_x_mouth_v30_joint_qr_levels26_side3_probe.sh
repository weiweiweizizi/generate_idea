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
  --levels=2,6 \
  --action_basis_init_path=scripts/lq/init_basis/basis_x_shared_2_6.npy \
  --side_basis_init_path=scripts/lq/init_basis/basis_x_side_from_level2.npy \
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
  --basis_orthogonalization=joint_global_qr \
  --discrete_side_loss_enabled=False \
  --epochs=50 \
  --batch_size=64 \
  --side_semantic_enabled=True \
  --side_basis_count=3 \
  --side_loss_weight=0.3 \
  --use_dataset_aux=False \
  --early_branch_factorization=True \
  --free_pool_size=2 \
  --side_pool_size=2 \
  --private_pool_size=1 \
  --side_pooling=fixed_region2_contrast \
  --side_z_dim=32 \
  --output_dir=outputs/lq_x_mouth_v30_joint_qr_levels26_side3_probe_win20_e50
