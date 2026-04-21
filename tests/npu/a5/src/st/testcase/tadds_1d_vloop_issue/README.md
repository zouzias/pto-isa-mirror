# TADDS 1D RV_VLOOP Hardware Loop Issue

## Problem

On A5, TADDS 1D generates RV_VLDS (scalar register offset) instead of RV_VLDI (immediate offset),
preventing the compiler from using RV_VLOOP hardware loops.

## Benchmark Results (1x4096 float32, A5 beta.2 dav_3510)

| Op    | Total Ticks | RV_VLOOP | RVECSU | Performance |
|-------|-------------|----------|--------|-------------|
| TADD  | 22,149      | ✅ Yes (vloop_id=1) | 50     | Baseline    |
| TADDS | 34,326      | ❌ No (vloop_id=0)  | 9,160  | **1.55x slower** |

TADDS generates **183x more scalar instructions** due to missing hardware loop optimization.

## Log Evidence

### TADD Baseline - Uses RV_VLOOP Hardware Loop

**EXU dump shows `vloop_id=1` (hardware loop enabled):**
```
[PERF] [1643] EXU RV_VADD type.F32 id=115 retire 1649 vloop_id 1 vloop_pc 10d0d510 ele_cnt 64
[PERF] [1644] EXU RV_VADD type.F32 id=120 retire 1650 vloop_id 1 vloop_pc 10d0d510 ele_cnt 64
[PERF] [1645] EXU RV_VADD type.F32 id=128 retire 1651 vloop_id 1 vloop_pc 10d0d510 ele_cnt 64
[PERF] [1646] EXU RV_VADD type.F32 id=133 retire 1652 vloop_id 1 vloop_pc 10d0d510 ele_cnt 64
...
```
Note: Consecutive cycles (1643→1644→1645...), minimal scalar overhead.

**ASU dump shows only 3 RV_SMOV per iteration:**
```
[1631] INSTR Retired. name=RV_SMOV id=107 ssid=1  pc=0x10D0D500
[1632] INSTR Retired. name=RV_SMOV id=108 ssid=2  pc=0x10D0D504
[1633] INSTR Retired. name=RV_SMOV id=109 ssid=3  pc=0x10D0D508
[1730] INSTR Retired. name=RV_SEND id=526 ssid=326 pc=0x10D0D528  ← barrier
```

### TADDS - No RV_VLOOP, Heavy Scalar Overhead

**EXU dump shows `vloop_id=0` (no hardware loop):**
```
[PERF] [1739] EXU RV_VADDS type.F32 id=138 retire 1745 vloop_id 0 vloop_pc 0 ele_cnt 64
[PERF] [1789] EXU RV_VADDS type.F32 id=156 retire 1795 vloop_id 0 vloop_pc 0 ele_cnt 64
[PERF] [1839] EXU RV_VADDS type.F32 id=174 retire 1845 vloop_id 0 vloop_pc 0 ele_cnt 64
[PERF] [1889] EXU RV_VADDS type.F32 id=192 retire 1895 vloop_id 0 vloop_pc 0 ele_cnt 64
...
```
Note: 50-cycle gap between VADDS instructions due to scalar loop overhead.

**ASU dump shows inner loop scalar sequence (per VADDS iteration):**
```
[1654] INSTR Retired. name=RV_SMOV     id=112 ssid=1  pc=0x10D0D400  ← move
[1663] INSTR Retired. name=RV_SCMP    id=117 ssid=6  pc=0x10D0D41C  ← compare
[1664] INSTR Retired. name=RV_SCBZI   id=118 ssid=7  pc=0x10D0D420  ← branch
[1683] INSTR Retired. name=RV_SMOV     id=120 ssid=9  pc=0x10D0D428  ← move
[1684] INSTR Retired. name=RV_SMOV     id=121 ssid=10 pc=0x10D0D42C  ← move
[1692] INSTR Retired. name=RV_SCMP    id=123 ssid=12 pc=0x10D0D450  ← compare
[1693] INSTR Retired. name=RV_SCBZI   id=124 ssid=13 pc=0x10D0D454  ← branch
[1712] INSTR Retired. name=RV_SZEROEXT id=126 ssid=15 pc=0x10D0D45C  ← zero extend
[1713] INSTR Retired. name=RV_SZEROEXT id=127 ssid=16 pc=0x10D0D460  ← zero extend
[1716] INSTR Retired. name=RV_SMUL    id=128 ssid=17 pc=0x10D0D464  ← multiply (addr calc)
[1717] INSTR Retired. name=RV_SMUL    id=129 ssid=18 pc=0x10D0D468  ← multiply (addr calc)
[1719] INSTR Retired. name=RV_SADD    id=130 ssid=19 pc=0x10D0D46C  ← add (addr calc)
[1720] INSTR Retired. name=RV_SCMP    id=131 ssid=20 pc=0x10D0D470  ← compare
[1721] INSTR Retired. name=RV_SMOV     id=132 ssid=21 pc=0x10D0D474  ← move
[1722] INSTR Retired. name=RV_SCBZI   id=133 ssid=22 pc=0x10D0D478  ← branch
... then VADDS executes ...
[1733] INSTR Retired. name=RV_SADDI   id=136 ssid=25 pc=0x10D0D43C  ← loop counter++
[1734] INSTR Retired. name=RV_SSUBI   id=137 ssid=26 pc=0x10D0D440  ← predicate calc
```

**Summary:** Each TADDS iteration requires ~14 scalar ops for loop control and address calculation,
compared to TADD which uses hardware vloop (vloop_id=1) with minimal scalar overhead.

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
grep -c "vloop_id 1" core0.veccore0.rvec.EXU.dump  # TADD: >0, TADDS: 0
grep -c "RV_S" core0.veccore0.rvec.ASU.dump        # TADDS has 183x more
```

### Expected Output
- Both tests pass (`[PASSED] 2 tests`)
- TADDS test runs ~1.5x longer than TADD baseline
- TADD shows `vloop_id 1`, TADDS shows `vloop_id 0`

## Root Cause

When a scalar operand is present in the operation (TADDS), the compiler generates
`RV_VLDS` (scalar register addressing) instead of `RV_VLDI` (immediate addressing).
This prevents the compiler from using `RV_VLOOP` hardware loop optimization, forcing
a software scalar loop with ~14 instructions per iteration.

## Dump Files

Pre-generated dumps are available in `dumps/` directory:
- `tadd_baseline/` - TADD test dumps (shows vloop_id=1)
- `tadds/` - TADDS test dumps (shows vloop_id=0)

Compressed versions in git:
- `core0.veccore0.instr_log.dump.gz` - instruction retire log
- `core0.veccore0.instr_popped_log.dump.gz` - instruction pop log
- `core0.veccore0.rvec.EXU.dump.gz` - execution unit log
- `core0.veccore0.rvec.ASU.dump.gz` - auxiliary scalar unit log

## Files

- `tadds_1d_vloop_issue_kernel.cpp` - Kernel with TADD and TADDS implementations
- `main.cpp` - GTest harness with in-memory data generation
- `gen_data.py` - Stub for run_st.py compatibility (no external data needed)
- `CMakeLists.txt` - Build configuration
- `dumps/` - Pre-generated simulator dump logs

## Environment

- **CANN:** 9.0.0-beta.2
- **Simulator:** dav_3510 (A5 full chip model)
- **Compiler:** ccec clang 15.0.5
