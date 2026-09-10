#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export DEX_YCB_DIR="${DEX_YCB_DIR:-$LIP_WORK_DIR/cache/raw_available_20260910}"
INDEX=cache/s0_available_20260910
OUT=runs/available_20260910
mkdir -p "$OUT"
python tools/overfit_small.py --config configs/dexycb_lip_v1.yaml --num-clips 32 --steps 500 --data-root "$DEX_YCB_DIR" --index-root "$INDEX" --allow-verified-subset --out "$OUT/overfit32" > "$OUT/overfit32.log" 2>&1
python tools/probe_batch.py --config configs/dexycb_lip_v1.yaml --data-root "$DEX_YCB_DIR" --index-root "$INDEX" --allow-verified-subset --out configs/candidate_available_20260910.yaml > "$OUT/probe.log" 2>&1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lip.train --config configs/candidate_available_20260910.yaml --data-root "$DEX_YCB_DIR" --index-root "$INDEX" --allow-verified-subset --max-steps 50 --force-rollout 4 --output "$OUT/ddp" > "$OUT/ddp_launcher.log" 2>&1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lip.train --config configs/candidate_available_20260910.yaml --data-root "$DEX_YCB_DIR" --index-root "$INDEX" --allow-verified-subset --max-steps 53 --force-rollout 4 --output "$OUT/ddp" --resume "$OUT/ddp/last.pt" > "$OUT/ddp_resume.log" 2>&1
python -m lip.evaluate --config configs/candidate_available_20260910.yaml --checkpoint "$OUT/overfit32/last.pt" --data-root "$DEX_YCB_DIR" --index-root "$INDEX" --allow-verified-subset --split val --mode one-step --max-frames 32 --out "$OUT/val_onestep" > "$OUT/val_onestep.log" 2>&1
python -m lip.evaluate --config configs/candidate_available_20260910.yaml --checkpoint "$OUT/overfit32/last.pt" --data-root "$DEX_YCB_DIR" --index-root "$INDEX" --allow-verified-subset --split val --mode closed-loop --limit-streams 20 --out "$OUT/val_closed" > "$OUT/val_closed.log" 2>&1
