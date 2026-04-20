# TADDS 1D RV_VLOOP Hardware Loop Issue

## Problem

On A5, TADDS 1D generates RV_VLDS (scalar register offset) instead of RV_VLDI (immediate offset),
preventing the compiler from using RV_VLOOP hardware loops.

## Benchmark Results (1x4096 float32, A5 beta.2 dav_3510)

| Op    | Total Ticks | RV_VLOOP | RVECSU | Performance |
|-------|-------------|----------|--------|-------------|
| TADD  | 22,205      | 10       | 50     | Baseline    |
| TADDS | 34,332      | 0        | 9,160  | **1.55x slower** |

TADDS generates **183x more scalar instructions** due to missing hardware loop optimization.

## Instruction Evidence

**TADD (uses RV_VLDI + RV_VLOOP):**
```
RV_VLDI Vd[0], Sn[65]=0x100, #offset=8, dist:NORMAL, #p=1
```

**TADDS (uses RV_VLDS, no RV_VLOOP):**
```
RV_VLDS Vd[0], Sn[8]=0x0, Sn2[9]=0x0, Sm[70]=0x100, dist:NORMAL, #p=0
```

## How to Reproduce

### Prerequisites
- CANN 9.0.0-beta.2 with A5 simulator (dav_3510)
- pto-isa repository

### Steps

```bash
# 1. Source CANN environment
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh

# 2. Run the test
python3 tests/script/run_st.py -r sim -v a5 -t tadds_1d_vloop_issue

# 3. Verify results - check instruction counts
cd tests/npu/a5/src/st/build/bin
grep -c RV_VLOOP core0.veccore0.rvec.ASU.dump   # TADD: ~10, TADDS: 0
grep -c RVECSU core0.veccore0.rvec.ASU.dump     # TADD: ~50, TADDS: ~9160
```

### Expected Output
- Both tests pass (`[PASSED] 2 tests`)
- TADDS test runs ~1.5x longer than TADD baseline
- Instruction logs confirm missing RV_VLOOP in TADDS

## Root Cause

When a scalar operand is present in the operation (TADDS), the compiler generates
`RV_VLDS` (scalar register addressing) instead of `RV_VLDI` (immediate addressing).
This prevents the compiler from using `RV_VLOOP` hardware loop optimization.

## Files

- `tadds_1d_vloop_issue_kernel.cpp` - Kernel with TADD and TADDS implementations
- `main.cpp` - GTest harness with in-memory data generation
- `gen_data.py` - Stub for run_st.py compatibility (no external data needed)
- `CMakeLists.txt` - Build configuration

## Environment

- **CANN:** 9.0.0-beta.2
- **Simulator:** dav_3510 (A5 full chip model)
- **Compiler:** ccec clang 15.0.5
