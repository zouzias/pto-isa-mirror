#!/bin/bash
# Analyze instruction dumps after running the test

DUMP_DIR=${1:-.}

echo "=== TADD vs TADDS Instruction Analysis ==="
echo ""

for test in TADD TADDS; do
    echo "--- ${test} ---"
    DUMP="${DUMP_DIR}/core0.veccore0.instr_log.dump"
    if [ -f "$DUMP" ]; then
        echo "RV_VLDI (immediate):  $(grep -c 'RV_VLDI' $DUMP)"
        echo "RV_VLDS (scalar reg): $(grep -c 'RV_VLDS' $DUMP)"
        echo "RV_VLOOP:             $(grep -c 'RV_VLOOP' $DUMP)"
        echo "RVECSU:               $(grep -c 'RVECSU' $DUMP)"
    else
        echo "Dump not found: $DUMP"
    fi
    echo ""
done
