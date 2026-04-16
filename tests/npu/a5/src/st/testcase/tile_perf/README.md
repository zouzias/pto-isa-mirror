# Tile Performance Benchmark (tile_perf)

Measure EPC (Elements Per Cycle) for PTO tile operations on A5 simulator.

## Supported Operations

| Type | Operations |
|------|------------|
| **Binary** | tadd, tsub, tmul, tdiv, tmax, tmin |
| **Unary** | texp, tlog, tsqrt, tabs, tneg, trcp, trsqrt |
| **Scalar** | tadds, tsubs, tmuls, tdivs, tmaxs, tmins |

## Quick Start

```bash
# 1. Edit test cases
cd ~/pto-isa/tests/npu/a5/src/st/testcase/tile_perf
vi input.csv

# 2. Build and run
cd ~/pto-isa
python3 tests/script/run_st.py -r sim -v a5 -t tile_perf

# 3. Run specific test
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/cann-9.0.0-alpha.1/aarch64-linux/simulator/Ascend910_9599/lib:~/pto-isa/tests/npu/a5/src/st/build/lib
cd ~/pto-isa/tests/npu/a5/src/st/build/bin
./tile_perf --gtest_filter="TADDBenchTest.case_float_32x64"
```

## input.csv Format

```csv
# op,dtype,tile_h,tile_w,valid_h,valid_w[,scalar]
tadd,float32,32,64,32,64
tadds,float32,32,64,32,64,3.14
texp,float32,1,2048,1,2048
```

| Column | Description |
|--------|-------------|
| op | Operation name (tadd, tadds, texp, etc.) |
| dtype | Data type: float32, float16, bfloat16, int32, int16, int8, uint8 |
| tile_h | Tile height (rows) |
| tile_w | Tile width (cols) |
| valid_h | Valid rows (<= tile_h) |
| valid_w | Valid cols (<= tile_w) |
| scalar | Scalar value for *s ops (optional, default 1.0) |

## Output Analysis

### Key Metrics

After running a test, extract metrics from dump files:

```bash
# VF execution time (per iteration)
grep "vf_real_execute_time" core0.veccore0.instr_log.dump

# ASU instruction breakdown
grep "instr_name" core0.veccore0.rvec.ASU.dump | cut -d" " -f6 | sort | uniq -c | sort -rn

# EXU instruction breakdown  
grep "instr_name" core0.veccore0.rvec.EXU.dump | cut -d" " -f6 | sort | uniq -c | sort -rn
```

### EPC Calculation

```
EPC = valid_rows × valid_cols / vf_real_execute_time
```

- Use **warm** VF cycles (iterations 3-10, skip first 2 for icache warmup)
- 10 iterations per test, use `pipe_barrier(PIPE_ALL)` between iterations

### Example Results (A5 Simulator)

| Op | Shape | Elements | Warm VF | EPC | ASU/VL | Notes |
|----|-------|----------|---------|-----|--------|-------|
| TADD | 32×64 | 2048 | 243 cy | 8.4 | 2.0 | 2D path |
| TADDS | 32×64 | 2048 | 242 cy | 8.5 | 4.0 | 2D path |
| TADD | 1×2048 | 2048 | 104 cy | 19.7 | ~1 | 1D path |
| TADDS | 1×2048 | 2048 | 1646 cy | 1.2 | 12+ | 1D path, ASU bound |

### ASU Instruction Analysis

**Healthy (TADD 2D):**
- RV_SMOV: loop bookkeeping
- RV_SNOP: pipeline sync

**ASU Bound (TADDS 1D):**
- RV_SZEROEXT: scalar type conversion
- RV_SMUL: scalar format conversion
- RV_SCMP/RV_SCBZI: loop control
- Indicates scalar handling overhead per VL

## Known Issues

1. **TADDS 1D path is slow** — scalar type conversion happens per-iteration instead of being hoisted
2. **2D vs 1D:** 1D tiles have better EPC for TADD but worse for TADDS
3. **First VF cold:** ~2-3x slower due to icache miss

## File Structure

```
tile_perf/
├── README.md           # This file
├── CMakeLists.txt      # Build config
├── input.csv           # Test case definitions
├── main.cpp            # Test framework
├── tile_perf_kernel.cpp # Kernel implementations
├── gen_data.py         # Generate input data
├── generate_code.py    # Auto-generate kernel from CSV
└── run_bench.py        # Parse results to CSV
```

## Adding New Operations

1. Add case to `input.csv`
2. If op needs new kernel template, edit `tile_perf_kernel.cpp`
3. Rebuild: `python3 tests/script/run_st.py -r sim -v a5 -t tile_perf`
