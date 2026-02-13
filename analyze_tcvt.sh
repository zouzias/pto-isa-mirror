#!/bin/bash
# Quick analyzer for TCVT saturation tests
# Usage: ./analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Find the build directory
BUILD_DIR="tests/npu/a5/src/st/build"
SCRIPT_PATH="tests/npu/a5/src/st/testcase/tcvt/analyze_saturation.py"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}❌ Error: Analysis script not found at $SCRIPT_PATH${NC}"
    exit 1
fi

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ Error: Build directory not found at $BUILD_DIR${NC}"
    exit 1
fi

# If no arguments, show usage and list available tests
if [ $# -eq 0 ]; then
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  TCVT SATURATION TEST ANALYZER (from project root)${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}📖 Usage:${NC}"
    echo "  ./analyze_tcvt.sh <test_case_name>"
    echo ""
    echo -e "${YELLOW}💡 Example:${NC}"
    echo "  ./analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32"
    echo ""
    echo -e "${YELLOW}🔗 Full workflow:${NC}"
    echo "  1. Run test:  python3 tests/script/run_st.py -r sim -v a5 -t tcvt -g TCVTTest.saturation_int32_int16_1x32"
    echo "  2. Analyze:   ./analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32"
    echo ""
    
    # List available test cases in build directory
    if ls "$BUILD_DIR"/TCVTTest.saturation_* >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Available saturation test cases in build directory:${NC}"
        echo "┌──────────────────────────────────────────────────────────────────────────────┐"
        i=1
        for dir in "$BUILD_DIR"/TCVTTest.saturation_*; do
            if [ -d "$dir" ]; then
                testname=$(basename "$dir")
                # Check if output files exist
                if [ -f "$dir/output_default.bin" ]; then
                    status="${GREEN}✓ HAS OUTPUT${NC}"
                else
                    status="${YELLOW}○ no output${NC}"
                fi
                printf "│  %2d. %-50s %s\n" "$i" "$testname" "$status"
                i=$((i+1))
            fi
        done | sed 's/$/                                     │/' | cut -c1-80
        echo "└──────────────────────────────────────────────────────────────────────────────┘"
    else
        echo -e "${YELLOW}⚠️  No saturation test cases found in build directory${NC}"
    fi
    echo ""
    exit 0
fi

TEST_CASE="$1"

# Check if test case directory exists
if [ ! -d "$BUILD_DIR/$TEST_CASE" ]; then
    echo -e "${RED}❌ Error: Test case directory not found: $BUILD_DIR/$TEST_CASE${NC}"
    echo ""
    echo -e "${YELLOW}Available test cases:${NC}"
    ls -d "$BUILD_DIR"/TCVTTest.saturation_* 2>/dev/null | xargs -n1 basename | head -10
    exit 1
fi

# Run the analysis
echo -e "${GREEN}🔍 Analyzing test case: ${TEST_CASE}${NC}"
echo -e "${GREEN}   Build directory: ${BUILD_DIR}${NC}"
echo ""

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ABS_BUILD_DIR="$SCRIPT_DIR/$BUILD_DIR"
ABS_SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH"

# Change to build directory and run analysis
cd "$ABS_BUILD_DIR" || exit 1
python3 "$ABS_SCRIPT_PATH" "$TEST_CASE"
exit_code=$?

# Return to original directory
cd - > /dev/null || exit 1

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Analysis complete${NC}"
else
    echo ""
    echo -e "${RED}❌ Analysis failed with exit code $exit_code${NC}"
fi

exit $exit_code
