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

---

## Extended Configurations (Configs 6–9)

In addition to the original 5 ping-pong baselines, the suite has been extended
with 4 additional kernel families that explore A-large tiles, B-only
N-buffering, half-N L0C splitting, and L1 buffer-id reuse for >32-id stress.

All extended configs share the same problem shape (M=32, K=1024, N=256,
fp16→fp32) and the same `A_gm.bin`/`B_gm.bin`/`golden.bin` data. All pass
with `bad count: 0` and `max diff ≈ 2.67e-05`.

### Config 6 — `RunCubeMatmul*ALarge*` (A-large baseline)

A held in L1 as a single big `[32, 128]` tile (ping-pong, 2 L1 slots), B
ping-pong as before. INNER_K = A_K_TILE / B_K_TILE views over the L1 A.

| Test | A L1 | B L1 | K_tile | INNER |
|------|------|------|--------|-------|
| `CubeMatmulNBufTest.buf4_aLarge_K16` | 2×8 KiB | 4×8 KiB | 16 | 8 |
| `CubeMatmulNBufTest.buf8_aLarge_K16` | 2×8 KiB | 8×8 KiB | 16 | 8 |
| `CubeMatmulNBufTest.buf4_aLargeK64`  | 2×8 KiB | 4×16 KiB| 64 | 2 |

### Config 7 — `RunCubeMatmulBNBuf` (B N-buffering, A fixed)

Generalises Config 6 by parameterising the B-side N-buffer count
independently of A. `INNER = 128 / B_K_TILE`.

| Test | N_BUFS_B | B_K | B L1 / L0B |
|------|----------|-----|------------|
| `bnbuf2_K16_8KB`   | 2 | 16 | 16 KiB |
| `bnbuf4_K16_8KB`   | 4 | 16 | 32 KiB |
| `bnbuf8_K16_8KB`   | 8 | 16 | 64 KiB (exact L0B fit) |
| `bnbuf2_K32_16KB`  | 2 | 32 | 32 KiB |
| `bnbuf4_K32_16KB`  | 4 | 32 | 64 KiB (exact L0B fit) |

### Config 8 — `RunCubeMatmulBNBufNSplit` (4 KiB half-N B tile)

B tile shrunk to `[B_K=16, N_HALF=128] = 4 KiB` per slot. Two independent L0C
accumulators `cTile[0]` (cols 0..127) and `cTile[1]` (cols 128..255) decouple
the matmul WAW chain across N halves. Same total GM→L1 bytes as Config 7
(MTE2-bw bound); just a different schedule with finer N-granularity.

Buffer-id layout: A `0..1`, B `2..2+N-1`, L0A `next 2`, L0B `next N`, L0C
`last 2` (one per N half).

| Test | N_BUFS_B | B L1 / L0B | Notes |
|------|----------|------------|-------|
| `bnbufnsplit2_K16_4KB`  | 2  | 8 KiB / 8 KiB   | too few B slots |
| `bnbufnsplit4_K16_4KB`  | 4  | 16 KiB / 16 KiB | sweet spot      |
| `bnbufnsplit8_K16_4KB`  | 8  | 32 KiB / 32 KiB | extra buf, same MTE2 |
| `bnbufnsplit16_K16_4KB` | 16 | 64 KiB / 64 KiB | uses ids ≥ 32   |

### Config 8b — `RunCubeMatmulBNBufNSplitR` (rearranged buf-id pool)

Same memory plan as Config 8 with `N_BUFS_B=16` (16 distinct L1+L0B slots),
but the **buffer-id allocation is rearranged** so the kernel fits in a 32-id
HW envelope:

```
A_L1 :  0,1                         (PIPE_MTE2)
L0C  :  2,3                         (PIPE_M / PIPE_FIX, one per half)
L0A  :  4,5                         (PIPE_MTE1 / PIPE_M)
B_L1 :  6 .. 6+P-1                  (PIPE_MTE2 / PIPE_MTE1)
L0B  :  6+P .. 6+2P-1               (PIPE_MTE1 / PIPE_M)
```

- Default (HW32): `P = min(N_BUFS_B, 13)` → uses ids 0..31.
  At the wrap point (`flat == P`) a `pipe_barrier(PIPE_ALL)` drains
  in-flight ops on the recycled id before re-acquiring.
- `-DPTO_BUFID_HW_GT32`: `P = N_BUFS_B`, no cap, no barrier.

| Test | N_BUFS_B | HW32 pool | barriers/outer |
|------|----------|-----------|----------------|
| `bnbufnsplitR16_K16_4KB` | 16 | 13 | 1 (at flat=13)  |

### Config 9 — `RunCubeMatmulBL1Reuse` (8 KiB full-N B tile, L1 id reuse)

Reverts to a full-N B tile `[B_K=16, N=256] = 8 KiB` (single L0C accumulator)
to amortise scalar overhead, while exercising **L1 buffer-id reuse**: L1 can
hold up to 32 distinct B slots (256 KiB / 8 KiB), but on 32-id HW the L1
buf-id pool is capped at 16 and recycles every 16 inner iterations. L0B
remains capped at 8 (64 KiB / 8 KiB) and uses a distinct id per slot
(no reuse).

```
A_L1 :  0,1
L0C  :  2
L0A  :  3,4
L0B  :  5..12                       (8 ids, no reuse)
B_L1 :  13..13+L1_POOL-1
```

- Default (HW32): `L1_POOL = min(N_BUFS_B_L1, 16)` → uses ids 0..28.
  When `N_BUFS_B_L1 > L1_POOL` the L1 b_id reuses with `flat % L1_POOL`
  and inserts `pipe_barrier(PIPE_ALL)` at each wrap (`flat % L1_POOL == 0`).
- `-DPTO_BUFID_HW_GT32`: `L1_POOL = N_BUFS_B_L1`, no cap, no barrier
  (e.g. 32-buf variant uses ids 13..44).

| Test | N_BUFS_B_L1 | N_BUFS_L0B | L1 id pool (HW32) | wrap barriers |
|------|-------------|------------|-------------------|----------------|
| `bl1reuse16_K16_8KB` | 16 | 8 | 16 (no reuse) | 0 |
| `bl1reuse32_K16_8KB` | 32 | 8 | 16 (reuse)    | 4 (every 16 of 64 inner iters) |

### Buffer-id semantics notes

The HW maintains a per-`(pipe, id)` event FIFO of bounded depth. Reuse of an
id is **correctness-safe** as long as every access is bracketed by matched
`get_buf` / `rls_buf`: it merges (over-approximates) dependencies, never
removes them. Real failure modes for reuse are limited to:

1. FIFO overflow on a single pipe → single-pipe stall (mitigated by the
   `pipe_barrier(PIPE_ALL)` at the wrap point).
2. Cross-pipe acquire-order cycles → true deadlock (avoided by keeping
   reuse within a single producer pipe at a time).
3. Asymmetric `get` / `rls` counts on data-dependent control flow → token
   leak (kept balanced in all configs above).
4. Bare (un-bracketed) accesses sharing a recycled id → real RAW hazard
   (none of the configs above contain bare accesses).

### Performance summary (sim model time, lower is better)

| Test | Model time (ms) |
|------|-----------------|
| `bnbufnsplit2_K16_4KB`   | 108.4 |
| `bnbufnsplit4_K16_4KB`   |  67.9 |
| `bnbufnsplit8_K16_4KB`   |  70.9 |
| `bnbufnsplit16_K16_4KB`  |  98.5 |
| `bnbufnsplitR16_K16_4KB` |  80.3 |
| `bl1reuse16_K16_8KB`     |  66.5 |
| `bl1reuse32_K16_8KB`     |  66.9 |

The 8 KiB tile (Config 9) wins overall by amortising scalar / get_buf-rls_buf
overhead; L1 id-reuse adds essentially zero overhead (<1%) versus the 16-buf
no-reuse baseline because the 4 wrap barriers are cheap relative to the
64 inner iterations.
