#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PTO_LIB_PATH="${PTO_LIB_PATH:-/home/llx/pto-isa}"

exec /usr/local/python3.11.13/bin/python3 "${SCRIPT_DIR}/benchmark.py" \
  --case aligned_2k_8e \
  --impl custom \
  --mode forward \
  --warmup 10 \
  --iters 50 \
  --json
