#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR to the extracted official dataset}"
python tools/verify_preflight.py --config configs/resolved_8gpu.yaml
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lip.train \
 --config configs/resolved_8gpu.yaml --data-root "$DEX_YCB_DIR" \
 --index-root cache/dexycb_s0 --output runs/lip_v1_s0 "$@"
