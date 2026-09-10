#!/usr/bin/env bash
set -euo pipefail
LIP_WORK_DIR="${LIP_WORK_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export LIP_WORK_DIR
cd "$LIP_WORK_DIR"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH="$LIP_WORK_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$LIP_WORK_DIR/cache/torch"
export TORCH_EXTENSIONS_DIR="$LIP_WORK_DIR/cache/torch_extensions"
export TMPDIR="$LIP_WORK_DIR/cache/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-4}"
mkdir -p "$TMPDIR" runs
