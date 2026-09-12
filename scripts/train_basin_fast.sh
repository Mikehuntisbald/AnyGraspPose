#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
source "$LIP_WORK_DIR/.venv-fp/bin/activate"
export PYTHONPATH="$LIP_WORK_DIR/runs/basin_speed_v1/candidate/src"
export OPENBLAS_NUM_THREADS=1
python - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('runs/basin_speed_v1');r=json.loads((p/'approval.json').read_text())
assert r['passed']
assert hashlib.sha256((p/'threaded.yaml').read_bytes()).hexdigest()==r['config_sha256']
h=hashlib.sha256();source=p/'candidate/src/lip'
for f in sorted(source.rglob('*.py')):
    h.update(str(f.relative_to(source)).encode());h.update(f.read_bytes())
assert h.hexdigest()==r['code_sha256'], 'Runtime source differs from tested checkpoint'
PY
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR}"
exec python -m torch.distributed.run --standalone --nproc_per_node=8 -m lip.train \
  --config runs/basin_speed_v1/threaded.yaml --data-root "$DEX_YCB_DIR" \
  --index-root cache/dexycb_s0 --output runs/lip_v1_s0 "$@"
