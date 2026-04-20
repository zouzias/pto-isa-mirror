# tile_perf — PTO Tile Operation Performance Benchmarks

## Overview

Cycle-accurate performance benchmarks for PTO tile operations on the A5 (Ascend950PR_9599) simulator.
Tests are driven by CSV input files and auto-generated via `generate_code.py`.

## Supported Operations

| Op | Category | Signature | Description |
|----|----------|-----------|-------------|
| `TADD` | Binary elementwise | `TADD(dst, src0, src1)` | Element-wise addition |
| `TEXP` | Unary elementwise | `TEXP(dst, src)` | Element-wise exponential |
| `TADDS` | Scalar elementwise | `TADDS(dst, src, scalar)` | Add scalar to every element |
| `TROWSUM` | Row reduce | `TROWSUM(dst, src)` | Sum across columns → H×1 column vector |
| `TCOLSUM` | Col reduce | `TCOLSUM(dst, src, tmp, false)` | Sum across rows → 1×W row vector |
| `TROWEXPAND` | Row broadcast | `TROWEXPAND(dst, src)` | Broadcast col-0 across all columns |
| `TCOLEXPAND` | Col broadcast | `TCOLEXPAND(dst, src)` | Broadcast row-0 across all rows |

### Data Types

- `float` (fp32) — 4 bytes per element
- `half` (fp16) — 2 bytes per element

### Constraints

- **Minimum width:** 256 (4 × VL=64) for proper vector utilization
- **TROWSUM fp32:** H ≥ 8 (ColMajor dst alignment: H × 4 ≥ 32 bytes)
- **TROWSUM fp16:** H ≥ 16 (ColMajor dst alignment: H × 2 ≥ 32 bytes)
- **Tile sizes tested:** 16KB, 32KB, 64KB

## Quick Start

```bash
# 1. Generate kernel code from CSV
cd ~/pto-isa/tests/npu/a5/src/st/testcase/tile_perf
python3 generate_code.py --csv input.csv

# 2. Build
cd ~/pto-isa/tests/npu/a5/src/st/build
make tile_perf -j8

# 3. Run all tests
./bin/tile_perf

# 4. Run filtered tests
./bin/tile_perf --gtest_filter="*TADD*"
./bin/tile_perf --gtest_filter="*TROWSUM*half*"
```

## Input CSV Format

```
# op,dtype,tile_h,tile_w,valid_h,valid_w[,scalar]
TADD,float,32,256,32,256
TADDS,half,64,256,64,256,1.5
TROWSUM,float,16,512,16,512
```

| Column | Description |
|--------|-------------|
| `op` | Operation name (TADD, TEXP, TADDS, TROWSUM, TCOLSUM, TROWEXPAND, TCOLEXPAND) |
| `dtype` | `float` or `half` |
| `tile_h` | Tile height (rows) |
| `tile_w` | Tile width (columns) |
| `valid_h` | Valid height (usually = tile_h) |
| `valid_w` | Valid width (usually = tile_w) |
| `scalar` | Scalar value for TADDS only (optional, default 1.5) |

### Example CSV Files

#### `input_elementwise.csv` — Elementwise ops (TADD, TEXP, TADDS)

54 tests: 3 ops × 2 dtypes × 3 sizes (16/32/64KB) × 3 shapes (1D, H×256, H×512)

```csv
# TADD fp32 — 32KB (8192 elements)
TADD,float,1,8192,1,8192
TADD,float,32,256,32,256
TADD,float,16,512,16,512

# TEXP fp16 — 32KB (16384 elements)
TEXP,half,1,16384,1,16384
TEXP,half,64,256,64,256
TEXP,half,32,512,32,512

# TADDS fp32 — 32KB (scalar=1.5)
TADDS,float,1,8192,1,8192,1.5
TADDS,float,32,256,32,256,1.5
TADDS,float,16,512,16,512,1.5
```

#### `input_reduce_bcast.csv` — Reduce & broadcast ops

70 tests: 4 ops × 2 dtypes × 3 sizes × 2–3 shapes (H×256, H×512, H×1024)

```csv
# TROWSUM fp32 — 32KB (H>=8 required)
TROWSUM,float,32,256,32,256
TROWSUM,float,16,512,16,512
TROWSUM,float,8,1024,8,1024

# TCOLSUM fp16 — 64KB
TCOLSUM,half,128,256,128,256
TCOLSUM,half,64,512,64,512
TCOLSUM,half,32,1024,32,1024

# TROWEXPAND fp32 — 16KB
TROWEXPAND,float,16,256,16,256
TROWEXPAND,float,8,512,8,512
TROWEXPAND,float,4,1024,4,1024

# TCOLEXPAND fp16 — 32KB
TCOLEXPAND,half,64,256,64,256
TCOLEXPAND,half,32,512,32,512
TCOLEXPAND,half,16,1024,16,1024
```

#### Custom single-op CSV

You can create a minimal CSV to test a single configuration:

```csv
# Just one TADD test
TADD,float,32,256,32,256
```

## EPC Extraction

EPC (Elements Per Cycle) measures compute throughput per VF (Vector Function) invocation.

| Field | Source | Description |
|-------|--------|-------------|
| `vf_real_execute_time` | `core0.veccore0.instr_log.dump` | Pure rvec compute cycles (use this) |
| `vf_execute_time` | same file | Total pop-to-retire (includes icache prefetch) |

```python
# Skip first cold VF (icache miss), average warm VFs
warm_times = [168, 168, 168, ...]  # vf_real_execute_time for VF 2+
avg_warm = sum(warm_times) / len(warm_times)
vf_epc = elements / avg_warm
# Example: 8192 / 296 = 27.68 EPC
```

## Regression Runner

```bash
cd ~/pto-isa/tests/npu/a5/src/st/testcase/tile_perf

# Run all tests with parallel execution and stall detection
python3 run_32kb_regression.py --parallel 5 --stall-timeout 60

# Run filtered tests
python3 run_32kb_regression.py --filter "*TROWSUM*"

# Results written to regression_results.csv
```

Output columns: `test_id, op, dtype, shape, elements, epc, vf_cycles, n_vf, tick, status, attempts`

## Files

| File | Purpose |
|------|---------|
| `generate_code.py` | Auto-generates `tile_perf_kernel.cpp` and `main.cpp` from input CSV |
| `input.csv` | Active input CSV (symlink or copy of one of the examples) |
| `input_elementwise.csv` | 54 elementwise test cases (TADD/TEXP/TADDS) |
| `input_reduce_bcast.csv` | 70 reduce/broadcast test cases |
| `input_full.csv` | Combined full test suite |
| `run_32kb_regression.py` | Parallel regression runner with EPC extraction |
| `regression_results.csv` | Latest regression results |
| `tile_perf_kernel.cpp` | Generated kernel code (do not edit manually) |
| `main.cpp` | Generated GTest harness (do not edit manually) |
| `worker.py` | Legacy regression runner with SQLite DB |
| `report.py` | A5 vs A2A3 comparison report generator |
| `schema.sql` | SQLite schema for legacy results DB |
| `REPORT.md` | Full performance results and analysis |
| `SKILL.md` | Skill documentation and coverage matrix |

## See Also

- [REPORT.md](REPORT.md) — Detailed EPC results and A5 vs A2A3 comparison
- [SKILL.md](SKILL.md) — Coverage matrix and skill documentation
