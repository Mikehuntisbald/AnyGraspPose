#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
source "$LIP_WORK_DIR/.venv-fp/bin/activate"
export PYTHONPATH="$LIP_WORK_DIR/runs/basin_boundary_v3/candidate/src"
export BASIN_CONFIG="$LIP_WORK_DIR/runs/basin_boundary_v3/candidate/configs/basin_validated.yaml"
python - <<'PY'
import pathlib,json,hashlib,os,yaml
p=pathlib.Path(os.environ['BASIN_CONFIG']);c=yaml.safe_load(p.read_text());r=json.loads(pathlib.Path('runs/basin_boundary_v3/preflight_passed.json').read_text())
assert c['basin_quality_policy']=='validated' and c['basin_preflight_passed'] and r['passed']
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['config_sha256']
PY
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR}"
exec python -m torch.distributed.run --standalone --nproc_per_node=8 -m lip.train \
 --config "$BASIN_CONFIG" --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 --output runs/lip_v1_s0 "$@"
