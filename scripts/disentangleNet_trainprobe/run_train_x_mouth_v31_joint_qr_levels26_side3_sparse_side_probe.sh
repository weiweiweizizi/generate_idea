#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

set +u
source /home/weizilin/anaconda3/etc/profile.d/conda.sh
conda activate dl
set -u

python scripts/disentangleNet_trainprobe/train.py \
  --data_roots=data/win20-step20/IMR,data/win20-step20/TT \
  --epochs=50 \
  --batch_size=64 \
  --output_dir=outputs/disentangleNet_trainprobe/v32_tri_region_masked_win20_e50
