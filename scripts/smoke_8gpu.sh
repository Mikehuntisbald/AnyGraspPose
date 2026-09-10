#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR to the extracted official dataset}"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lip.train \
 --config configs/resolved_8gpu.yaml --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 \
 --output runs/ddp_real --max-steps 50 --force-rollout 4 2>&1 | tee runs/ddp_real_launcher.log
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lip.train \
 --config configs/resolved_8gpu.yaml --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 \
 --output runs/ddp_real --max-steps 53 --force-rollout 4 --resume runs/ddp_real/last.pt 2>&1 | tee runs/ddp_real_resume.log
