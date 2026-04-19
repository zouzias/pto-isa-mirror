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
# Build
cd ~/pto-isa/tests/npu/a5/src/st/build
make tile_perf -j8

# Run all tests
./bin/tile_perf

# Run specific tests
./bin/tile_perf --gtest_filter="*TADD*"
```

## Key Files

| File | Purpose |
|------|---------|
| `main.cpp` | GTest harness with 54 test cases |
| `tile_perf_kernel.cpp` | TADD/TADDS/TEXP kernel implementations |
| `worker.py` | Regression runner with EPC extraction |
| `report.py` | Comparison report generator |
| `schema.sql` | SQLite schema for results database |
| `input.csv` | Test case definitions |
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

| Op | Dtype | Sizes | Shapes |
|----|-------|-------|--------|
| TADD | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |
| TEXP | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |
| TADDS | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |

**Total:** 54 tests (3 ops × 2 dtypes × 3 sizes × 3 shapes)
**Width requirement:** ≥ 256 (4×VL) for proper vector pipeline stress

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

| Metric | A5 | A2A3 |
|--------|-----|------|
| TADD EPC | ~24-30 | ~48-59 |
| TADDS EPC | **1.0-1.3** (broken) | ~48-59 |
| Shape sensitivity | 1D faster | Shape-agnostic |
| EPC source | vf_real_execute_time | pop→retire |

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
