#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

python3 -m pypto emit "$ROOT/demo/pyPTO/vec_add.py" -o "$ROOT/demo/pto/vec_add.pto"
python3 -m ptoas "$ROOT/demo/pto/vec_add.pto" -o "$ROOT/demo/bin/vec_add.bin" --soc a3 --run-mode npu
mkdir -p "$ROOT/demo/out"
"$ROOT/demo/bin/vec_add.bin" --out-dir "$ROOT/demo/out"
