#!/bin/bash
# --------------------------------------------------------------------------------
# clean_workspace.sh - remove generated workspace artifacts under
# kernels/automode/a2a3/. Targets:
#   - run_log/                       (created by run_all.sh)
#   - <kernel>/prof/                 (created by -p / --profile in run.sh / run_all.sh)
#   - <family>/<subkernel>/prof/     (for MoE / MoEv2 sub-kernels)
#
# Does NOT touch per-kernel build/ directories: each kernel's run.sh already
# does `rm -rf build` before rebuild, so build/ is owned by that flow.
#
# Usage:
#   bash clean_workspace.sh              # remove
#   bash clean_workspace.sh --dry-run    # print what would be removed, do nothing
#   bash clean_workspace.sh -h|--help
# --------------------------------------------------------------------------------

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1;;
    esac
done

targets=()

# Family-level run_log/.
if [[ -d "${HERE}/run_log" ]]; then
    targets+=("${HERE}/run_log")
fi

# Per-kernel prof/ at depths 2 (top-level kernels) and 3 (MoE/MoEv2 sub-kernels).
while IFS= read -r d; do
    [[ -n "${d}" ]] && targets+=("${d}")
done < <(find "${HERE}" -mindepth 2 -maxdepth 3 -type d -name prof 2>/dev/null)

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "[clean] nothing to remove"
    exit 0
fi

for t in "${targets[@]}"; do
    rel="${t#${HERE}/}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[clean] would remove ${rel}"
    else
        echo "[clean] removing ${rel}"
        rm -rf "${t}"
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[clean] dry-run complete; rerun without --dry-run to actually delete"
fi
