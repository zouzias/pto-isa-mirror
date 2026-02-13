#!/bin/bash
# Run ALL saturation tests and analyze results
# Usage: ./run_all_saturation_tests.sh [runtime] [version]

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
RUNTIME="${1:-sim}"
VERSION="${2:-a5}"
TESTCASE="tcvt"

# All saturation test cases
SATURATION_TESTS=(
    "TCVTTest.saturation_fp16_int8_1x32"
    "TCVTTest.saturation_fp16_int16_1x32"
    "TCVTTest.saturation_fp16_uint8_1x32"
    "TCVTTest.saturation_fp32_int16_1x32"
    "TCVTTest.saturation_int32_int16_1x32"
    "TCVTTest.saturation_int64_int32_1x32"
)

# Track results
declare -a PASSED_TESTS
declare -a FAILED_TESTS
TOTAL_TESTS=${#SATURATION_TESTS[@]}
CURRENT_TEST=0

echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  RUN ALL SATURATION TESTS - Batch Mode${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Runtime:       ${RUNTIME}${NC}"
echo -e "${GREEN}Version:       ${VERSION}${NC}"
echo -e "${GREEN}Total tests:   ${TOTAL_TESTS}${NC}"
echo ""
echo -e "${YELLOW}📋 Test cases to run:${NC}"
for test in "${SATURATION_TESTS[@]}"; do
    echo "   • $test"
done
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Run each test
for test_name in "${SATURATION_TESTS[@]}"; do
    CURRENT_TEST=$((CURRENT_TEST + 1))
    
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  TEST [$CURRENT_TEST/$TOTAL_TESTS]: ${test_name}${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Run the test
    echo -e "${CYAN}🔧 Running test...${NC}"
    python3 tests/script/run_st.py -r "$RUNTIME" -v "$VERSION" -t "$TESTCASE" -g "$test_name" > /tmp/test_output_$$.log 2>&1
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}❌ Test FAILED (exit code: $TEST_EXIT_CODE)${NC}"
        FAILED_TESTS+=("$test_name")
        echo ""
        echo -e "${YELLOW}Showing last 20 lines of test output:${NC}"
        tail -20 /tmp/test_output_$$.log
        echo ""
        continue
    fi
    
    echo -e "${GREEN}✅ Test PASSED${NC}"
    echo ""
    
    # Analyze the results
    echo -e "${CYAN}📊 Analyzing results...${NC}"
    ./analyze_tcvt.sh "$test_name" "$VERSION" > /tmp/analyze_output_$$.log 2>&1
    ANALYZE_EXIT_CODE=$?
    
    if [ $ANALYZE_EXIT_CODE -ne 0 ]; then
        echo -e "${YELLOW}⚠️  Analysis had issues (exit code: $ANALYZE_EXIT_CODE)${NC}"
        FAILED_TESTS+=("$test_name (analysis)")
    else
        # Check if output matches golden
        if grep -q "Match rate:.*100.00%" /tmp/analyze_output_$$.log; then
            echo -e "${GREEN}✅ Analysis PASSED - 100% match rate${NC}"
            PASSED_TESTS+=("$test_name")
        else
            echo -e "${RED}❌ Analysis FAILED - Results don't match golden${NC}"
            FAILED_TESTS+=("$test_name (mismatch)")
        fi
    fi
    
    echo ""
    
    # Show brief summary from analysis
    echo -e "${CYAN}📋 Quick summary:${NC}"
    grep -A 5 "SPECIAL VALUES ANALYSIS" /tmp/analyze_output_$$.log | head -10 || echo "  (No special values analysis available)"
    echo ""
    
    # Small delay between tests
    sleep 1
done

# Clean up temp files
rm -f /tmp/test_output_$$.log /tmp/analyze_output_$$.log

# Final summary
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  FINAL SUMMARY${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Total tests run:    ${TOTAL_TESTS}${NC}"
echo -e "${GREEN}Passed:             ${#PASSED_TESTS[@]}${NC}"
echo -e "${RED}Failed:             ${#FAILED_TESTS[@]}${NC}"
echo ""

if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Passed tests:${NC}"
    for test in "${PASSED_TESTS[@]}"; do
        echo "   ✓ $test"
    done
    echo ""
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "   ✗ $test"
    done
    echo ""
    echo -e "${YELLOW}💡 To re-run a specific failed test:${NC}"
    echo "   ./run_and_analyze_tcvt.sh <test_name>"
    echo ""
    echo -e "${YELLOW}💡 To analyze a specific test:${NC}"
    echo "   ./analyze_tcvt.sh <test_name>"
    echo ""
fi

# Set exit code based on results
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  🎉 ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ⚠️  SOME TESTS FAILED${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    exit 1
fi
