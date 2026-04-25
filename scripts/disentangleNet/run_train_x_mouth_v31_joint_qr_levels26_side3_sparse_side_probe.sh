#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

set +u
source /home/weizilin/anaconda3/etc/profile.d/conda.sh
conda activate dl
set -u

python scripts/disentangleNet/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --epochs=50 \
  --batch_size=64 \
  --output_dir=outputs/disentangleNet/v31_joint_qr_levels26_side3_sparse_side_probe_win20_e50
