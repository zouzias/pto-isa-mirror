---
name: tile-perf-check
description: A5 vs A2A3 tile operation performance benchmarking and regression testing for PTO-ISA. Use for analyzing TADD/TADDS/TEXP EPC metrics, debugging VF scalar issues, and running comparative benchmarks.
---

# Tile Performance Benchmark Skill

Performance benchmarking and regression testing for PTO tile operations (TADD, TADDS, TEXP) comparing A5 vs A2A3 architectures.

## Project Context

**Repository:** `gitcode.com/ChanKaLok/pto-isa`
**Branch:** `tile-perf-check`
**Working Directory:** `~/pto-isa/tests/npu/a5/src/st/testcase/tile_perf/`

## Quick Start

```bash
# Generate code from a CSV
cd ~/pto-isa/tests/npu/a5/src/st/testcase/tile_perf
python3 generate_code.py --csv input.csv                # elementwise ops
python3 generate_code.py --csv input_reduce_bcast.csv   # reduce/broadcast ops

# Build
cd ~/pto-isa/tests/npu/a5/src/st/build
make tile_perf -j8

# Run all tests
./bin/tile_perf

# Run specific tests
./bin/tile_perf --gtest_filter="*TADD*"
./bin/tile_perf --gtest_filter="*TROWSUM*"
```

## Key Files

| File | Purpose |
|------|---------|
| `main.cpp` | GTest harness (auto-generated) |
| `tile_perf_kernel.cpp` | Kernel implementations (auto-generated) |
| `generate_code.py` | Single-source code generator from CSV |
| `input.csv` | Active elementwise test cases (TADD/TEXP/TADDS) |
| `input_reduce_bcast.csv` | Reduce & broadcast test cases (TROWSUM/TCOLSUM/TROWEXPAND/TCOLEXPAND) |
| `input_elementwise.csv` | Backup: elementwise-only (54 cases) |
| `input_full.csv` | Backup: all ops combined |
| `worker.py` | Regression runner with EPC extraction |
| `report.py` | Comparison report generator |
| `schema.sql` | SQLite schema for results database |
| `regression_results.csv` | Latest A5 32KB regression results |
| `REPORT.md` | Performance comparison findings |

## EPC Extraction

### A5 Method (VF-based)

**Source:** `core0.veccore0.instr_log.dump`
**Field:** `vf_real_execute_time` (pure VF compute time)

```python
# Skip first cold VF (icache miss), average warm VFs
warm_times = [168, 168, 168, ...]  # vf_real_execute_time for VF 2+
avg_warm = sum(warm_times) / len(warm_times)
vf_epc = elements / avg_warm
# Example: 4096 / 168 = 24.38 EPC
```

### A2A3 Method (pop→retire)

**Sources:**
- `core0.veccore0.instr_popped_log.dump` (dispatch cycles)
- `core0.veccore0.instr_log.dump` (retire cycles)

```python
# First VEC pop → last VEC retire per iteration
# Skip first 2 cold iterations
epc = elements / (last_retire - first_pop)
```

## Test Matrix

### Elementwise (input.csv / input_elementwise.csv)

| Op | Dtype | Sizes | Shapes |
|----|-------|-------|--------|
| TADD | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |
| TEXP | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |
| TADDS | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |

**Total:** 54 tests (3 ops × 2 dtypes × 3 sizes × 3 shapes)

### Reduce & Broadcast (input_reduce_bcast.csv)

| Op | Dtype | Sizes | Shapes |
|----|-------|-------|--------|
| TROWSUM | float/half | 16KB, 32KB, 64KB | Hx256, Hx512, Hx1024 |
| TCOLSUM | float/half | 16KB, 32KB, 64KB | Hx256, Hx512, Hx1024 |
| TROWEXPAND | float/half | 16KB, 32KB, 64KB | Hx256, Hx512, Hx1024 |
| TCOLEXPAND | float/half | 16KB, 32KB, 64KB | Hx256, Hx512, Hx1024 |

**Total:** 70 tests (TROWSUM has 16 due to ColMajor dst alignment: fp32 H≥8, fp16 H≥16)
**Note:** TROWSUM dst is `Tile<Vec, T, H, 1, ColMajor>` — requires `H*sizeof(T) ≥ 32`

**Width requirement:** ≥ 256 (4×VL) for proper vector pipeline stress

## Coverage Status

| Arch | Framework | Status | Notes |
|------|-----------|--------|-------|
| A5 | PTO | ✅ 32KB elementwise done | 18/18 PASS, results in regression_results.csv |
| A5 | PTO | 🔲 reduce/broadcast | 70 tests built, ready to run |
| A5 | PTO | 🔲 16KB/64KB elementwise | Not yet run |
| A2A3 | PTO | 🔲 Not started | Needs CANN alpha.1 env, different EPC extraction |
| A5 | AscendC | 🔲 Not started | Needs AscendC kernel implementations |

## A5 PTO 32KB Elementwise Results

All 18 tests PASS (Ascend950PR_9599, CANN 9.0.0-beta.2 simulator, 10 VF iterations):

### TADD (Binary Add) — avg EPC = 36.0

| Dtype | Shape | Elems | VF_EPC | VF_cy | Tick |
|-------|-------|-------|--------|-------|------|
| fp32 | 1×8192 | 8192 | 27.68 | 296 | 38836 |
| fp32 | 16×512 | 8192 | 27.31 | 300 | 39128 |
| fp32 | 32×256 | 8192 | 24.24 | 338 | 38593 |
| fp16 | 1×16384 | 16384 | 55.35 | 296 | 38695 |
| fp16 | 32×512 | 16384 | 48.47 | 338 | 38864 |
| fp16 | 64×256 | 16384 | 32.90 | 498 | 38544 |

### TEXP (Unary Exp) — avg EPC = 27.1

| Dtype | Shape | Elems | VF_EPC | VF_cy | Tick |
|-------|-------|-------|--------|-------|------|
| fp32 | all 3 shapes | 8192 | 25.76 | 318 | ~24k |
| fp16 | all 3 shapes | 16384 | 28.49 | 575 | ~24k |

### TADDS (Scalar Add) — avg EPC = 1.64 ⚠️

| Dtype | Shape | Elems | VF_EPC | VF_cy | Tick |
|-------|-------|-------|--------|-------|------|
| fp32 | 1×8192 | 8192 | 1.31 | 6254 | 65307 |
| fp32 | 16×512 | 8192 | 1.15 | 7109 | 73859 |
| fp32 | 32×256 | 8192 | 1.02 | 8021 | 83031 |
| fp16 | 1×16384 | 16384 | 2.62 | 6254 | 65188 |
| fp16 | 32×512 | 16384 | 2.04 | 8021 | 83042 |
| fp16 | 64×256 | 16384 | 1.67 | 9785 | 100644 |

### Key Observations

- **TADD/TEXP**: Healthy EPC (25–55). fp16 gets ~2× elements in similar cycles → higher EPC. Flat 1D shapes fastest.
- **TADDS**: Confirmed A5 bug — EPC ~1–2.6, **15–27× worse** than TADD. Wider tiles (more rows) = worse.
- **TEXP is shape-insensitive**: All fp32 shapes give identical 318 cycles, all fp16 give 575 cycles.

## Known Issues

### TADDS A5 Performance Bug (CRITICAL)

**Symptom:** A5 TADDS is 37-58x slower than A2A3
**Root Cause:** Scalar broadcast generates excessive RVECSU ops

From `core0.veccore0.rvec.ASU.dump`:
```
RV_SCMP:     670  # loop bound check
RV_SCBZI:    670  # conditional branch
RV_SZEROEXT: 640  # scalar type conversion
RV_SMUL:     640  # scalar format conversion
```

**A2A3 uses:** Hardware VF repeat directly (`vadds(dst, src, scalar, repeats, ...)`)
**A5 uses:** C++ for-loop → compiled to RVECSU scalar control

**Fix needed:** A5 VF-fused `vadds` intrinsic (like A2A3)

**Related files:**
- `include/pto/npu/a5/TBinSOp.hpp` (A5 TADDS implementation)
- `include/pto/npu/a2a3/TAddS.hpp` (A2A3 reference)

## Running Regression

```bash
# Full A5 regression
cd ~/pto-isa/tests/npu/a5/src/st/testcase/tile_perf
python3 worker.py --arch a5 --tests "*" --output /tmp/pto_regress

# A2A3 regression
python3 worker.py --arch a2a3 --tests "*" --output /tmp/pto_regress

# Specific tests with debug
python3 worker.py --arch a5 --tests "TADD_float_*" --debug

# Generate report
python3 report.py results.db --baseline a2a3
```

## LightRAG Knowledge Graph

Query PTO-ISA knowledge base for architecture details:

```bash
curl -s -X POST http://192.168.0.106:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "A5 VF EPC extraction vf_real_execute_time", "mode": "hybrid", "top_k": 5}'
```

**Topics available:**
- PTO instructions (TLOAD, TSTORE, TADD, TADDS, TCVT, etc.)
- A2A3/A5 architecture differences
- Memory model (UB, L1, L0A/L0B/L0C)
- VPTO surface ISA (vlds, vsts, vadd, vmuls)
- VF pipeline, EPC metrics

## Architecture Comparison Summary

| Metric | A5 (PTO) | A2A3 (PTO) | A5 (AscendC) |
|--------|----------|------------|--------------|
| TADD EPC (32KB fp32) | 24–28 | TBD | TBD |
| TADD EPC (32KB fp16) | 33–55 | TBD | TBD |
| TEXP EPC (32KB fp32) | 25.76 | TBD | TBD |
| TADDS EPC (32KB fp32) | **1.0–1.3** (broken) | ~48–59 | TBD |
| Shape sensitivity | 1D faster | Shape-agnostic | TBD |
| EPC source | vf_real_execute_time | pop→retire | TBD |

## CANN Environment

```bash
# A5 (beta.2)
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
export SOC_VERSION=Ascend950PR_9599
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/aarch64-linux/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH

# A2A3
source /usr/local/Ascend/cann-9.0.0-alpha.1/set_env.sh
export SOC_VERSION=Ascend910B1
```

## Git Workflow

```bash
# Current branch status
git log -1 --oneline
# Expected: 5efc9be5 tile_perf: Add A5 vs A2A3 regression worker and report

# Push changes
git add .
git commit -m "tile_perf: <description>"
git push origin tile-perf-check
```

## Next Steps

1. **Fix TADDS:** Implement A5 VF-fused `vadds` intrinsic
2. **Expand matrix:** Add TMUL, TSUB, TMAX, TMIN operations
3. **Add fp16 A2A3:** Complete half-precision comparison
4. **Automate:** CI integration for regression on commit

## References

- `npu_skills/pto-isa/verification/a5-epc-extraction.md` — EPC extraction guide
- `npu_skills/pypto/ptoas-insert-sync-analysis.md` — Sync insertion analysis
- `benchmarks/a5_epc_reference.md` — A5 EPC baselines
- `benchmarks/a2a3_epc_reference.md` — A2A3 EPC baselines
