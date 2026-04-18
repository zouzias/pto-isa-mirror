# TADDS RV_VLOOP Hardware Loop Issue

## Problem Description

On A5 (Ascend910B3/Ascend950PR_9599), TADDS generates significantly more scalar unit 
instructions (RVECSU) compared to TADD, resulting in ~1.6x slower performance for 
identical data sizes.

## Root Cause

The compiler generates different load/store instruction types:

| Operation | Load Instruction | Store Instruction | Hardware Loop |
|-----------|-----------------|-------------------|---------------|
| **TADD**  | RV_VLDI (immediate offset) | RV_VSTI | RV_VLOOP ✅ |
| **TADDS** | RV_VLDS (scalar register)  | RV_VSTS | None ❌ |

When a scalar operand is present in the vector operation, the compiler:
1. Uses RV_VLDS instead of RV_VLDI for loads
2. Cannot emit RV_VLOOP hardware loop instructions
3. Falls back to RVECSU scalar loop control (9160 vs 50 instructions)

## Benchmark Results (A5 Simulator, beta.2)

Test case: 1x4096 float32, 10 iterations

| Metric | TADD | TADDS | Ratio |
|--------|------|-------|-------|
| Total Ticks | 21,799 | 34,253 | 1.57x slower |
| RV_VLOOP | 10 | 0 | - |
| RVECSU | 50 | 9,160 | 183x more |
| EPC | 1.88 | 1.20 | 0.64x |

## Expected Behavior

TADDS should achieve similar performance to TADD since:
1. The scalar operand is loop-invariant
2. Load addresses follow the same pattern as TADD
3. Hardware should be able to use RV_VLDI + RV_VLOOP

## Reproduction

-- The C compiler identification is GNU 15.2.0
-- The CXX compiler identification is GNU 15.2.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /home/linuxbrew/.linuxbrew/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /home/linuxbrew/.linuxbrew/bin/g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring incomplete, errors occurred!

## Environment

- CANN: 9.0.0-beta.2
- SOC: Ascend910B3 (A5)
- Simulator: Ascend950PR_9599 (dav_3510)
- Compiler: ccec clang version 15.0.5
