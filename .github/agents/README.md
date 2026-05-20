# PTO ISA Agents Guide

This document describes the agents in .github/agents/ and how to use them.

## Quick Start
- Use Master Orchestrator to pick the right agent when unsure.
- Use Testcase Author to create new ST testcases.
- Use Testcase Runner to execute testcases and log performance.
- Use Testcase Comparator to compare a2a3 vs a5 results.

## Agent: Master Orchestrator
Purpose
- Chooses the appropriate agent(s) and the order to run them.

When to use
- You are not sure which agent to call.
- The task requires multiple steps (create then run then compare).

What to provide
- Task goal
- Target SoC (a2a3 or a5)
- Constraints or files to modify

Typical outputs
- Which agent(s) are selected and the sequence.

Example prompt
- "Create a tadd testcase for a5 and then run it in sim."

## Agent: Testcase Author
Purpose
- Creates complete PTO ST testcases end-to-end.

Creates
- CMakeLists.txt
- *_kernel.cpp
- main.cpp
- gen_data.py
- README.md in the testcase folder (explains composition and mapping)

What to provide
- Operator or composite behavior
- Global, tile, valid shapes
- Dtypes
- Any scalars or constants
- Target SoC (a5 default)

Output locations
- tests/npu/<soc>/src/st/testcase/<testcase>/...

Example prompt
- "Create a testcase for TADD on a5. Use global 64x64, tile 64x64, valid 64x64, dtype float and half."

Notes
- Uses only existing PTO intrinsics.
- Adds a README.md explaining compound ops and mapping to intrinsics.

## Agent: Testcase Runner
Purpose
- Runs testcases and logs performance metrics.

How it runs
- Activates environment:
  - source /usr/local/Ascend/cann/set_env.sh
  - source .venv/bin/activate
- Uses:
  - python3 tests/script/run_st.py -r [sim|npu] -v [a2a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]

Metrics
- a2a3: system total ticks from core0_summary_log
- a5: rvec_veccore0_simd_busy_cycle or cube.cube_cubecore0_mac_busy_cycle

Output
- Appends to tests/npu/<testcase>_output.csv
- CSV header:
  testcase,gtest,soc,dtype,global_rows,global_cols,tile_rows,tile_cols,valid_rows,valid_cols,ticks_or_cycles,inputsize,throughput

What to provide
- SoC (a2a3 or a5)
- run mode (sim or npu)
- testcase name
- gtest filter(s), including suite prefix (for example: TADDTest.case_float_64x64_64x64)

Example prompt
- "Run tadd on a5 in sim with gtest TADDTest.case_float_64x64_64x64_64x64_64x64."

Notes
- Parses and appends results after each individual test case.
- Applies minimal compile fixes if needed, without changing core logic.

## Agent: Testcase Comparator
Purpose
- Compares a2a3 vs a5 throughput and writes a report.

How it matches
- Matches on testcase, dtype, global, tile, and valid shapes.
- Ignores gtest names for matching.

Output
- Report path: tests/npu/<testcase>__comparison.md

What to provide
- testcase name (for example: tadd)
- optional path to <testcase>_output.csv

Report contents
- Summary counts of where a3 or a5 is better
- Top 5 wins for each
- Table of all matched shapes with ratios
- Interpretation of trends
- Unmatched shapes

Example prompt
- "Compare tadd_output.csv and generate the report."

## File Locations
- Agents live in .github/agents/
- Outputs:
  - Testcases: tests/npu/<soc>/src/st/testcase/<testcase>/
  - CSV: tests/npu/<testcase>_output.csv
  - Comparison: tests/npu/<testcase>__comparison.md
