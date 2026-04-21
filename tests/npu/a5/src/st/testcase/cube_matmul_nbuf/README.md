# cube_matmul_nbuf — N-Buffer Cube MatMul Benchmark

## Purpose

Benchmarks the **ping-pong / N-buffer pattern** for cube matrix multiply on the
A5 NPU (Ascend 950B / Ascend950PR_9599 simulator).

The goal is to understand how different buffer depths (2, 4, 8) and tile sizes
(K_tile=16 vs K_tile=32) interact with MTE2 (GM→L1) bandwidth and scalar
overhead.  Results and analysis are in [REPORT.md](REPORT.md).

**Matrix dimensions (all configs):** M=32, K=1024, N=256, fp16 → fp32

| Config | K_tile | N_Bufs | K_ITERS | L1 IDs | L0 IDs | Total IDs |
|--------|--------|--------|---------|--------|--------|-----------|
| buf2_ktile16_8KB  | 16 | 2 | 64 | 0–1  | 2–3   | 5  |
| buf4_ktile16_8KB  | 16 | 4 | 64 | 0–3  | 4–7   | 9  |
| buf8_ktile16_8KB  | 16 | 8 | 64 | 0–7  | 8–15  | 17 |
| buf2_ktile32_16KB | 32 | 2 | 32 | 0–1  | 2–3   | 5  |
| buf4_ktile32_16KB | 32 | 4 | 32 | 0–3  | 4–7   | 9  |

---

## GM Memory Layout — Row-Major

A, B, C are stored in standard row-major layout:

```
A : shape [M=32,  K=1024], stride [K, 1],   dtype fp16
B : shape [K=1024, N=256], stride [N, 1],   dtype fp16
C : shape [M=32,  N=256],  stride [N, 1],   dtype fp32
```

The kernel iterates over K in tiles of K_tile:

```
for k in range(K_ITERS):
    A_tile = A[0:M, k*K_tile : (k+1)*K_tile]   # shape [M, K_tile]
    B_tile = B[k*K_tile : (k+1)*K_tile, 0:N]   # shape [K_tile, N]
    C += A_tile @ B_tile
```

A single `A_gm.bin` / `B_gm.bin` / `golden.bin` is shared by all 5 configs.

### Buffer slot addresses (TASSIGN)

| Memory | K_tile=16 stride | K_tile=32 stride |
|--------|-----------------|-----------------|
| L1 A   | 0x800  (2 KB)   | 0x1000  (4 KB)  |
| L1 B   | 0x2000 (8 KB)   | 0x4000 (16 KB)  |
| L0A    | 0x400  (1 KB)   | 0x800   (2 KB)  |
| L0B    | 0x2000 (8 KB)   | 0x4000 (16 KB)  |

---

## Files

| File | Description |
|------|-------------|
| `cube_matmul_nbuf_kernel.cpp` | Kernel — 5 templated launch functions |
| `main_nbuf.cpp` | GTest harness — 5 test cases |
| `gen_data.py` | Golden data generator (row-major layout) |
| `CMakeLists.txt` | Build target `cube_matmul_nbuf` |
| `REPORT.md` | Full performance analysis and pipeline diagrams |
| `profiling/` | msprof trace.json files + SVG pipeline diagrams |

---

## Building and Running

### Prerequisites

```bash
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
```

### Build

```bash
cd tests/npu/a5/src/st
python3 ../../script/run_st.py -r sim -v a5 -t cube_matmul_nbuf --build-only
# or full clean build:
python3 ../../script/run_st.py -r sim -v a5 -t cube_matmul_nbuf
```

### Run a single config

```bash
# e.g. buf4_ktile16_8KB
python3 tests/script/run_st.py -r sim -v a5 -t cube_matmul_nbuf \
    -g CubeMatmulNBufTest.buf4_ktile16_8KB
```

Available test filter names:
- `CubeMatmulNBufTest.buf2_ktile16_8KB`
- `CubeMatmulNBufTest.buf4_ktile16_8KB`
- `CubeMatmulNBufTest.buf8_ktile16_8KB`
- `CubeMatmulNBufTest.buf2_ktile32_16KB`
- `CubeMatmulNBufTest.buf4_ktile32_16KB`

Expected: all 5 PASS (max diff < 1e-2, bad count = 0).

---

## Profiling with msprof

### Generate trace.json

```bash
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH

CFG=buf4_ktile16_8KB
mkdir -p /tmp/msprof_${CFG}
chmod 700 /tmp/msprof_${CFG}

cd tests/npu/a5/src/st/build/bin
export CAMODEL_LOG_PATH=$(pwd)/../CubeMatmulNBufTest.${CFG}
mkdir -p $CAMODEL_LOG_PATH

msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --output=/tmp/msprof_${CFG} \
  ./cube_matmul_nbuf --gtest_filter=CubeMatmulNBufTest.${CFG}
```

Output directory: `/tmp/msprof_<CFG>/OPPROF_<timestamp>/`
- `simulator/trace.json` — Chrome-format trace (open in `chrome://tracing`)
- `simulator/core0.cubecore0/core0.cubecore0_instr_exe.csv` — per-pipe cycle counts

### Generate SVG pipeline diagram

```bash
# Single config — generates pipeline_0_6us.svg in profiling/<cfg>/
python3 /tmp/gen_svg.py   # see profiling/ directory for the script

# Compare two configs side-by-side
python3 /tmp/compare_k16_k32.py
```

Pre-generated SVGs are in the `profiling/` directory:

| SVG | Contents |
|-----|----------|
| `profiling/pipeline_comparison.svg` | buf4_K16 vs buf8_K16 (0–6.5 µs) |
| `profiling/pipeline_k16_vs_k32.svg` | buf4_K16 vs buf4_K32 (0–9 µs) |

### Reading the SU perf summary log

After a sim run, the scalar PMU counters are in:

```
build/CubeMatmulNBufTest.<config>/core0.cubecore0_su_perf_summary_log
```

Key fields:

| Field | What it tells you |
|-------|-------------------|
| `su_issue_icache_miss_cycle` | ICache miss stall cycles (code too large → reduce code size) |
| `su_issue_ld_miss_hazard_cycle` | DCache load miss cycles |
| `su_issue_mte2_payloadQ_stall_cycle` | Scalar blocked waiting for MTE2 queue (MTE2-bound indicator) |
| `su_issue_branch_stall_cycle` | Branch prediction stalls |
| `su_busy_cycle` | Total scalar active cycles |

For buf4_K16: `mte2_payloadQ_stall_cycle = 17,176` — the dominant stall, confirming MTE2-bound behaviour.
