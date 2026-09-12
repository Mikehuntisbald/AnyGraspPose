#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
source "$LIP_WORK_DIR/.venv-fp/bin/activate"
export PYTHONPATH="$LIP_WORK_DIR/runs/basin_memory_fix/candidate/src"
python - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('runs/basin_memory_fix');r=json.loads((p/'preflight.json').read_text())
assert r['passed'] and hashlib.sha256((p/'config.yaml').read_bytes()).hexdigest()==r['config_sha256']
PY
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR}"
exec python -m torch.distributed.run --standalone --nproc_per_node=8 -m lip.train \
 --config runs/basin_memory_fix/config.yaml --data-root "$DEX_YCB_DIR" \
 --index-root cache/dexycb_s0 --output runs/lip_v1_s0 "$@"
