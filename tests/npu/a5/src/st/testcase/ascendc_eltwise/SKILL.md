---
name: ascendc-eltwise-regression
description: AscendC eltwise Add kernel baseline for A5 EPC regression vs PTO TADD. Covers FLAT (whole-tile Add) and ROWLOOP (row-by-row Add) variants, 10 shapes (fp32/fp16, 32KB tiles).
---

# AscendC Eltwise Regression Skill

## Purpose

Provide AscendC Add kernel baselines to compare against PTO TADD EPC results.
Answers: does AscendC VF-fused `Add(tile, tile, count)` match PTO TADD, or does row-loop decomposition hurt performance?

## Environment Setup

```bash
# A5 CANN (beta.2)
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
export SOC_VERSION=Ascend950PR_9599

# AscendC compiler (ASC language)
export PATH=/usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/compiler/ccec_compiler/bin:$PATH
```

## Build & Run

```bash
# Build via PTO ST framework
cd ~/pto-isa
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
python3 tests/script/run_st.py -r sim -v a5 -t ascendc_eltwise

# Or manual cmake
cd ~/pto-isa/tests/npu/a5/src/st/build
make ascendc_eltwise -j8
./bin/ascendc_eltwise
```

## Test Matrix

| Test Name | Variant | dtype | Shape | Elems | PTO TADD EPC (ref) |
|-----------|---------|-------|-------|-------|---------------------|
| flat_fp32_1x8192   | FLAT    | fp32  | 1x8192  | 8192  | 27.68 |
| flat_fp32_16x512   | FLAT    | fp32  | 16x512  | 8192  | 27.31 |
| flat_fp32_32x256   | FLAT    | fp32  | 32x256  | 8192  | 24.24 |
| flat_fp16_1x16384  | FLAT    | fp16  | 1x16384 | 16384 | 55.35 |
| flat_fp16_32x512   | FLAT    | fp16  | 32x512  | 16384 | 48.47 |
| flat_fp16_64x256   | FLAT    | fp16  | 64x256  | 16384 | 32.90 |
| rowloop_fp32_16x512  | ROWLOOP | fp32 | 16x512  | 8192  | 27.31 |
| rowloop_fp32_32x256  | ROWLOOP | fp32 | 32x256  | 8192  | 24.24 |
| rowloop_fp16_32x512  | ROWLOOP | fp16 | 32x512  | 16384 | 48.47 |
| rowloop_fp16_64x256  | ROWLOOP | fp16 | 64x256  | 16384 | 32.90 |

## Kernel Variants

### FLAT (whole-tile Add)
```cpp
Add(zLocal, xLocal, yLocal, ELEMENTS);  // ROWS*COLS in one call
```
- Single `Add` over entire flattened tile
- Hardware may fuse rows via VF repeat
- Expected: close to PTO TADD EPC

### ROWLOOP (row-by-row Add)
```cpp
for (uint32_t row = 0; row < ROWS; row++) {
    Add(zLocal[off], xLocal[off], yLocal[off], COLS);
}
```
- Per-row `Add`, no cross-row VF fusion
- Expected: slower for 2D tiles vs FLAT; same as FLAT for 1D

## EPC Extraction

After running, dumps are in `tests/npu/a5/src/st/build/bin/`

```bash
# Extract vf_real_execute_time values
grep "vf_real_execute_time" core0.veccore0.instr_log.dump

# Quick EPC calc
python3 << 'PYEOF'
import re
content = open('core0.veccore0.instr_log.dump').read()
times = [int(x) for x in re.findall(r'vf_real_execute_time=(\d+)', content)]
warm = times[1:]  # skip first cold VF
elements = 8192   # change per test shape
print(f'VF times: {warm}')
if warm:
    avg_cy = sum(warm) / len(warm)
    print(f'avg VF cycles = {avg_cy:.1f}')
    print(f'avg EPC = {elements / avg_cy:.2f}')
PYEOF
```

## Pipeline Pattern (TPipe/TQue)

```
CopyIn:  DataCopy(xLocal, xGm, N)   -> EnQue  (mark data ready)
Compute: xLocal = inQueueX.DeQue()  -> Add(z,x,y,N) -> outQueueZ.EnQue(z)
CopyOut: z = outQueueZ.DeQue()      -> DataCopy(zGm,z,N) -> FreeTensor
```

Key rules:
- `AllocTensor` only reserves UB memory, does NOT wait for DMA
- `EnQue` after `DataCopy` marks buffer ready for consumer
- `DeQue` blocks until async op completes
- Always pair `AllocTensor` + `FreeTensor`

## Expected Findings

| Scenario | Expected |
|----------|----------|
| FLAT 1D (1xN) | Approx PTO TADD EPC (same VF count, no row overhead) |
| FLAT 2D (HxW) | Close to PTO if compiler fuses; worse if not |
| ROWLOOP 2D (HxW) | Slower than PTO TADD (H separate VF dispatches vs fused) |

## Files

| File | Purpose |
|------|---------|
| `ascendc_eltwise_kernel.cpp` | AscendC kernels (FLAT + ROWLOOP variants, .cpp using kernel_operator.h) |
| `main.cpp` | GTest harness with precision check + ACL setup |
| `gen_data.py` | Stub for run_st.py compatibility |
| `CMakeLists.txt` | `pto_vec_st(ascendc_eltwise)` |
| `SKILL.md` | This file |

## ST Environment on pto-b10

```bash
# ST test runner
python3 tests/script/run_st.py -r sim -v a5 -t <testcase_name>

# CANN environments
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh   # A5 beta.2
source /usr/local/Ascend/cann-9.0.0-alpha.1/set_env.sh             # A2A3 alpha.1

# ASC compiler arch flags
# --npu-arch=dav-3510   (A5, Ascend950PR_9599)
# --npu-arch=dav-2201   (A2A3, Ascend910B1)

# Simulator lib path (needed for runtime)
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/aarch64-linux/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH

# Note: pto_vec_st CMake macro handles ASC compilation automatically
# It picks up --npu-arch from the ST framework's toolchain
```

## cann_skills References on pto-b10

```bash
~/cann_skills/ops/skills/ascendc-direct-invoke-template/   # kernel直调 template + run.sh
~/cann_skills/ops/skills/ascendc-api-best-practices/       # TPipe/TQue best practices
~/cann_skills/ops/skills/ascendc-code-review/              # perf review guide
~/cann_skills/ops/skills/ascendc-npu-arch/                 # arch flags reference
~/ascendc_bench/                                           # standalone AscendC bench (A2A3 verified)
~/ascendc_bench/v1/                                        # unified PTO vs AscendC compare framework
```

## PTO TADD Reference Results (from tile_perf/REPORT.md)

A5 32KB (Ascend950PR_9599, CANN 9.0.0-beta.2):

| Dtype | Shape   | Elems  | VF_EPC | VF_cy |
|-------|---------|--------|--------|-------|
| fp32  | 1x8192  | 8192   | 27.68  | 296   |
| fp32  | 16x512  | 8192   | 27.31  | 300   |
| fp32  | 32x256  | 8192   | 24.24  | 338   |
| fp16  | 1x16384 | 16384  | 55.35  | 296   |
| fp16  | 32x512  | 16384  | 48.47  | 338   |
| fp16  | 64x256  | 16384  | 32.90  | 498   |

## LightRAG Knowledge Base

```bash
curl -s -X POST http://192.168.0.106:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "AscendC Add VF EPC DataCopy TPipe TQue A5", "mode": "hybrid", "top_k": 5}'
```

## Results (fill in after running)

| Test | Variant | EPC | vs PTO TADD | Notes |
|------|---------|-----|-------------|-------|
| flat_fp32_1x8192   | FLAT    | TBD | TBD | |
| flat_fp32_16x512   | FLAT    | TBD | TBD | |
| flat_fp32_32x256   | FLAT    | TBD | TBD | |
| flat_fp16_1x16384  | FLAT    | TBD | TBD | |
| flat_fp16_32x512   | FLAT    | TBD | TBD | |
| flat_fp16_64x256   | FLAT    | TBD | TBD | |
| rowloop_fp32_16x512 | ROWLOOP | TBD | TBD | |
| rowloop_fp32_32x256 | ROWLOOP | TBD | TBD | |
| rowloop_fp16_32x512 | ROWLOOP | TBD | TBD | |
| rowloop_fp16_64x256 | ROWLOOP | TBD | TBD | |

## Next Steps

1. Run: `python3 tests/script/run_st.py -r sim -v a5 -t ascendc_eltwise`
2. Extract EPC from `build/bin/core0.veccore0.instr_log.dump`
3. Fill in Results table above
4. Compare FLAT vs ROWLOOP to confirm VF row-fusion hypothesis
5. Add AscendC EPC column to tile_perf/REPORT.md
