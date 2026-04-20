#!/bin/bash
# PTOAS Small Tile ST - Full Flow Script
# This script runs the complete flow:
# 1. Compile .pto IR with PTOAS to CCE C++
# 2. Build CPU simulator test
# 3. Run regression test
#
# Usage: ./run_ptoas_st.sh [--clean] [--debug] [--skip-compile]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTCASE_DIR="${SCRIPT_DIR}"
PTO_ISA_ROOT="${SCRIPT_DIR}/../../../.."
PTOAS_BIN="${PTOAS_BIN:-$HOME/PTOAS-official/build/tools/ptoas/ptoas}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
CLEAN=false
DEBUG=false
SKIP_COMPILE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --skip-compile)
            SKIP_COMPILE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--debug] [--skip-compile]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== PTOAS Small Tile ST Flow ===${NC}"
echo "Testcase: cube_matmul_4buf"
echo "PTOAS: ${PTOAS_BIN}"
echo ""

# Step 1: Check PTOAS exists
if [[ ! -f "${PTOAS_BIN}" ]]; then
    echo -e "${RED}ERROR: PTOAS not found at ${PTOAS_BIN}${NC}"
    echo "Set PTOAS_BIN environment variable to point to ptoas binary"
    exit 1
fi

# Step 2: Compile .pto to CCE C++ with PTOAS
if [[ "${SKIP_COMPILE}" == "false" ]]; then
    echo -e "${YELLOW}[Step 1/4] Compiling PTO IR with PTOAS...${NC}"
    
    PTO_INPUT="${TESTCASE_DIR}/ptoas_input/cube_matmul_4buf.pto"
    CCE_OUTPUT="${TESTCASE_DIR}/ptoas_output/cube_matmul_4buf_ptoas.cpp"
    
    mkdir -p "${TESTCASE_DIR}/ptoas_output"
    
    DEBUG_LEVEL=1
    if [[ "${DEBUG}" == "true" ]]; then
        DEBUG_LEVEL=2
    fi
    
    echo "  Input:  ${PTO_INPUT}"
    echo "  Output: ${CCE_OUTPUT}"
    
    # PTOAS level2 with auto-sync insertion
    ${PTOAS_BIN} \
        --pto-arch=a5 \
        --pto-level=level2 \
        --enable-insert-sync \
        --pto-insert-sync-debug=${DEBUG_LEVEL} \
        "${PTO_INPUT}" \
        -o "${CCE_OUTPUT}" \
        2>&1 | tee "${TESTCASE_DIR}/ptoas_output/compile.log"
    
    if [[ -f "${CCE_OUTPUT}" ]]; then
        echo -e "${GREEN}  PTOAS compilation successful${NC}"
    else
        echo -e "${RED}  PTOAS compilation failed - no output file${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[Step 1/4] Skipping PTOAS compilation (--skip-compile)${NC}"
fi

# Step 3: Generate test data
echo -e "${YELLOW}[Step 2/4] Generating test data...${NC}"
cd "${TESTCASE_DIR}"

if [[ ! -d "CubeMatmul4BufTest.case_f16_32x1024_1024x256" ]] || [[ "${CLEAN}" == "true" ]]; then
    rm -rf CubeMatmul4BufTest.*
    python3 gen_data.py
    echo -e "${GREEN}  Test data generated${NC}"
else
    echo "  Using existing test data"
fi

# Step 4: Build CPU simulator test
echo -e "${YELLOW}[Step 3/4] Building CPU simulator test...${NC}"

BUILD_ARGS=""
if [[ "${CLEAN}" == "true" ]]; then
    BUILD_ARGS="--clean"
fi

cd "${PTO_ISA_ROOT}"
python3 tests/run_cpu.py --testcase cube_matmul_4buf ${BUILD_ARGS} --build-only 2>&1 | tail -20

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}  Build successful${NC}"
else
    echo -e "${RED}  Build failed${NC}"
    exit 1
fi

# Step 5: Run regression test
echo -e "${YELLOW}[Step 4/4] Running regression test...${NC}"

cd "${PTO_ISA_ROOT}"
python3 tests/run_cpu.py --testcase cube_matmul_4buf --gtest_filter='CubeMatmul4BufTest.*' 2>&1 | tee "${TESTCASE_DIR}/test_result.log"

# Check result
if grep -q "PASSED" "${TESTCASE_DIR}/test_result.log"; then
    echo ""
    echo -e "${GREEN}=== TEST PASSED ===${NC}"
    grep "max diff" "${TESTCASE_DIR}/test_result.log" || true
else
    echo ""
    echo -e "${RED}=== TEST FAILED ===${NC}"
    exit 1
fi

echo ""
echo "Output files:"
echo "  - ptoas_output/cube_matmul_4buf_ptoas.cpp  (PTOAS-generated CCE C++)"
echo "  - ptoas_output/compile.log                  (PTOAS compile log)"
echo "  - test_result.log                           (CPU simulator test result)"
