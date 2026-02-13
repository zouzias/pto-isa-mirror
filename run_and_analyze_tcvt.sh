#!/bin/bash
# Run TCVT test and analyze results in one command
# Usage: ./run_and_analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
RUNTIME="sim"
VERSION="a5"
TESTCASE="tcvt"

if [ $# -eq 0 ]; then
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  RUN & ANALYZE TCVT SATURATION TESTS${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}📖 Usage:${NC}"
    echo "  ./run_and_analyze_tcvt.sh <test_case_name> [runtime] [version]"
    echo "  ./run_and_analyze_tcvt.sh all [runtime] [version]"
    echo ""
    echo -e "${YELLOW}💡 Examples:${NC}"
    echo "  ./run_and_analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32"
    echo "  ./run_and_analyze_tcvt.sh TCVTTest.saturation_fp16_int8_1x32 sim a5"
    echo "  ./run_and_analyze_tcvt.sh all                    # Run ALL saturation tests"
    echo ""
    echo -e "${YELLOW}📝 Default values:${NC}"
    echo "  Runtime: sim"
    echo "  Version: a5"
    echo ""
    echo -e "${YELLOW}🔗 What this script does:${NC}"
    echo "  1. Runs: python3 tests/script/run_st.py -r sim -v a5 -t tcvt -g <test_case>"
    echo "  2. Then: ./analyze_tcvt.sh <test_case>"
    echo ""
    
    # List available saturation tests from build directory
    BUILD_DIR="tests/npu/a5/src/st/build"
    if [ -d "$BUILD_DIR" ] && ls "$BUILD_DIR"/TCVTTest.saturation_* >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Available saturation test cases (in build):${NC}"
        echo "┌──────────────────────────────────────────────────────────────────────────────┐"
        i=1
        for dir in "$BUILD_DIR"/TCVTTest.saturation_*; do
            if [ -d "$dir" ]; then
                printf "│  %2d. %-70s │\n" "$i" "$(basename "$dir")"
                i=$((i+1))
            fi
        done
        echo "└──────────────────────────────────────────────────────────────────────────────┘"
    else
        echo -e "${YELLOW}💡 Saturation test cases:${NC}"
        echo "  • TCVTTest.saturation_fp16_int8_1x32"
        echo "  • TCVTTest.saturation_fp16_int16_1x32"
        echo "  • TCVTTest.saturation_fp16_uint8_1x32"
        echo "  • TCVTTest.saturation_fp32_int16_1x32"
        echo "  • TCVTTest.saturation_int32_int16_1x32"
        echo "  • TCVTTest.saturation_int64_int32_1x32"
    fi
    echo ""
    exit 0
fi

TEST_NAME="$1"
RUNTIME="${2:-sim}"
VERSION="${3:-a5}"

# Check if user wants to run all tests
if [ "$TEST_NAME" = "all" ] || [ "$TEST_NAME" = "ALL" ]; then
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Running ALL Saturation Tests${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    exec ./run_all_saturation_tests.sh "$RUNTIME" "$VERSION"
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 1: Running Test${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Test:    ${TEST_NAME}${NC}"
echo -e "${GREEN}Runtime: ${RUNTIME}${NC}"
echo -e "${GREEN}Version: ${VERSION}${NC}"
echo ""

# Run the test
python3 tests/script/run_st.py -r "$RUNTIME" -v "$VERSION" -t "$TESTCASE" -g "$TEST_NAME"
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Test execution failed with exit code $TEST_EXIT_CODE${NC}"
    echo -e "${YELLOW}⚠️  Proceeding with analysis for debugging...${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  STEP 2: Analyzing Results${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Small delay to ensure files are written
sleep 1

# Run the analysis
./analyze_tcvt.sh "$TEST_NAME" "$VERSION"
ANALYZE_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ TEST FAILED (exit code $TEST_EXIT_CODE) - Analysis completed for debugging${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    exit $TEST_EXIT_CODE
elif [ $ANALYZE_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ Analysis failed with exit code $ANALYZE_EXIT_CODE${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    exit $ANALYZE_EXIT_CODE
else
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ COMPLETE: Test executed and analyzed successfully${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    exit 0
fi
