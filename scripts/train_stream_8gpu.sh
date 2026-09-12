#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DEX_YCB_DIR:?Set DEX_YCB_DIR to the extracted official dataset}"
stream_python="${LIP_STREAM_PYTHON:-$PWD/.venv-fp/bin/python}"
if [[ ! -x "$stream_python" ]]; then
  echo "Set LIP_STREAM_PYTHON to the existing project Python environment" >&2
  exit 1
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$stream_python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m lip.train_stream --data-root "$DEX_YCB_DIR" "$@"
