#!/bin/bash
# Quick reference for TCVT saturation test scripts

cat << 'EOF'
╔════════════════════════════════════════════════════════════════════════════╗
║                    TCVT SATURATION TEST SCRIPTS                            ║
║                      Quick Reference Guide                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

📁 Available Scripts (from project root):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  Run and analyze ONE test:
    ./run_and_analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32
    
2️⃣  Run and analyze ALL tests (batch mode):
    ./run_and_analyze_tcvt.sh all
    
3️⃣  Just analyze (after running test manually):
    ./analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32
    
4️⃣  Batch mode with custom runtime/version:
    ./run_all_saturation_tests.sh sim a5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Available Saturation Tests:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • TCVTTest.saturation_fp16_int8_1x32    (FP16 → INT8)
  • TCVTTest.saturation_fp16_int16_1x32   (FP16 → INT16)
  • TCVTTest.saturation_fp16_uint8_1x32   (FP16 → UINT8)
  • TCVTTest.saturation_fp32_int16_1x32   (FP32 → INT16)
  • TCVTTest.saturation_int32_int16_1x32  (INT32 → INT16)
  • TCVTTest.saturation_int64_int32_1x32  (INT64 → INT32)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Common Workflows:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 Quick Test (One test):
   ./run_and_analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32

🔥 Run Everything (All 6 tests):
   ./run_and_analyze_tcvt.sh all

📊 Re-analyze existing results:
   ./analyze_tcvt.sh TCVTTest.saturation_int32_int16_1x32

🔍 List available tests:
   ./analyze_tcvt.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📖 Full Documentation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   See TCVT_WORKFLOW.md for complete documentation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
