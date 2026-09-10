#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR to the extracted official dataset}"
python tools/audit_data.py --data-root "$DEX_YCB_DIR" --out runs/audit
python tools/build_index.py --data-root "$DEX_YCB_DIR" --setup s0 --out cache/dexycb_s0
python tools/check_geometry.py --data-root "$DEX_YCB_DIR" --index cache/dexycb_s0 --num-samples 32 --out runs/geometry
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 tools/build_sampling.py --data-root "$DEX_YCB_DIR" --index cache/dexycb_s0 --length 8
python tools/merge_sampling.py --index cache/dexycb_s0 --world 8
python -m pytest -q 2>&1 | tee runs/pytest_preflight.log
python tools/overfit_small.py --config configs/dexycb_lip_v1.yaml --num-clips 32 --steps 500
python tools/probe_batch.py --config configs/dexycb_lip_v1.yaml --out configs/resolved_8gpu.yaml
bash scripts/smoke_8gpu.sh
python tools/verify_preflight.py --config configs/resolved_8gpu.yaml --approve
