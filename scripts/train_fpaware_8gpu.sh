#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
source "$LIP_WORK_DIR/.venv-fp/bin/activate"
export PYTHONPATH="${FP_AWARE_SOURCE_ROOT:-$LIP_WORK_DIR/runs/fpaware_19000/candidate}/src"
FP_AWARE_CONFIG="${FP_AWARE_CONFIG:-$LIP_WORK_DIR/runs/fpaware_19000/candidate/configs/fpaware_19000.yaml}"
export FP_AWARE_CONFIG
python - <<'PY'
import hashlib,json,os,pathlib,yaml
p=pathlib.Path(os.environ['FP_AWARE_CONFIG']);c=yaml.safe_load(p.read_text())
r=json.loads(pathlib.Path('runs/fpaware_19000/preflight_passed.json').read_text())
assert c['fpaware_preflight_passed'] and r['passed']
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['config_sha256']
PY
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR}"
exec python -m torch.distributed.run --standalone --nproc_per_node=8 -m lip.train \
 --config "$FP_AWARE_CONFIG" --data-root "$DEX_YCB_DIR" --index-root cache/dexycb_s0 --output runs/lip_v1_s0 "$@"
