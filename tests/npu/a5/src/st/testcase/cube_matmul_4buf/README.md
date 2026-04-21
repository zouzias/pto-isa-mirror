# cube_matmul_4buf Test Case

4-buffer software-pipelined matmul kernel demonstrating explicit synchronization techniques
on the A5 (Ascend950PR) architecture. Used as a reference case for studying `get_buf`/`rls_buf`
buffer-ID synchronization vs. `set_flag`/`wait_flag` event-ID synchronization.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| M (rows A / rows C) | 32 |
| K tile | 16 |
| K_ITERS | 64 |
| Total K | 1024 |
| N (cols B / cols C) | 256 |
| Input dtype | fp16 |
| Output dtype | fp32 |
| Pipeline depth | 4 buffers |

---

## Hardware Memory Hierarchy (A5)

| Level | Capacity | Used by |
|-------|----------|---------|
| L1 | 512 KB | aMatTile, bMatTile (staging from GM) |
| L0A | 64 KB | aTile (LeftTile — A operand to CUBE) |
| L0B | 64 KB | bTile (RightTile — B operand to CUBE) |
| L0C | 256 KB | cTile (AccTile — accumulator output) |

### Our 4-buffer Allocation

| Level | A tiles | B tiles | Total |
|-------|---------|---------|-------|
| L1 | 4 × 1 KB = 4 KB | 4 × 8 KB = 32 KB | ~36 KB / 512 KB ✅ |
| L0A | 4 × 1 KB = 4 KB | — | 4 KB / 64 KB ✅ |
| L0B | — | 4 × 8 KB = 32 KB | 32 KB / 64 KB ✅ |
| L0C | — | — | 32 KB / 256 KB ✅ |

### L1 Address Map

| Slot | aMatTile | bMatTile |
|------|----------|----------|
| 0 | 0x0000 | 0x10000 |
| 1 | 0x0800 | 0x12000 |
| 2 | 0x1000 | 0x14000 |
| 3 | 0x1800 | 0x16000 |

### L0 Address Map

| Slot | aTile (L0A) | bTile (L0B) |
|------|-------------|-------------|
| 0 | 0x000 | 0x0000 |
| 1 | 0x400 | 0x2000 |
| 2 | 0x800 | 0x4000 |
| 3 | 0xC00 | 0x6000 |

---

## Pipeline Stages and Data Flow

```
GM
 │  MTE2  (TLOAD: GM → L1)
 ▼
L1   ← aMatTile[0..3] / bMatTile[0..3]
 │  MTE1  (TMOV: L1 → L0A/L0B)
 ▼
L0A / L0B   ← aTile[0..3] / bTile[0..3]
 │  CUBE / PIPE_M  (TMATMUL / TMATMUL_ACC)
 ▼
L0C   ← cTile (AccTile)
 │  FIXPIPE / PIPE_FIX  (TSTORE: AccTile → GM)
 ▼
GM
```

**Key:** `TSTORE` of an `AccTile` uses **FIXPIPE** (`PIPE_FIX`), not MTE3.

---

## Synchronization Methods

### Final Results (A5 PEM Simulator, 2026-04-21)

| Kernel file | Sync method | Cycles | Max diff | Status |
|-------------|-------------|--------|----------|--------|
| `cube_matmul_4buf_kernel_barrier.cpp` | `pipe_barrier(PIPE_ALL)` | 52,957 | 9.54e-06 | ✅ PASS |
| `cube_matmul_4buf_preload_kernel.cpp` | `set_flag`/`wait_flag` | 24,276 | 9.54e-06 | ✅ PASS |
| `cube_matmul_4buf_kernel.cpp` (v3) | `get_buf`/`rls_buf` | 24,316 | 9.54e-06 | ✅ PASS |

### Method 1: pipe_barrier — Simple but slow

```cpp
pipe_barrier(PIPE_ALL);  // stall all pipelines
```

Correct but ~2× slower. Good for initial correctness validation.

### Method 2: set_flag / wait_flag — Recommended baseline

```cpp
set_flag(PIPE_MTE2, PIPE_MTE1, event_id);    // MTE2 signals L1 written
wait_flag(PIPE_MTE2, PIPE_MTE1, event_id);   // MTE1 waits for L1 data
```

Directional, per-pair, 8 IDs per pair. Requires priming (pre-issuing N-1 flags before loop).
Budget: 4 buffers × 2 IDs = 8 per directed pair — exactly fits.

### Method 3: get_buf / rls_buf — Explicit buffer tokens ✅

The most expressive method. Uses a global pool of 32 buffer IDs with acquire/release semantics.
A single ID handles both RAW (forward) and WAR (reverse) dependencies.

#### Correct pipe pairings for cube matmul

| Memory level | Tiles | IDs | Producer pipe | Consumer pipe |
|---|---|---|---|---|
| L1 | aMatTile, bMatTile | 0–3 | `PIPE_MTE2` | `PIPE_MTE1` |
| L0A / L0B | aTile, bTile | 4–7 | `PIPE_MTE1` | `PIPE_M` |
| L0C | cTile (AccTile) | 8 | `PIPE_M` | `PIPE_FIX` |

#### Sync pattern (per K iteration)

```cpp
int id_l1 = k % NUM_BUFS;          // L1 slot: 0,1,2,3
int id_l0 = id_l1 + L0_ID_OFFSET;  // L0 slot: 4,5,6,7

// MTE2: GM → L1
get_buf(PIPE_MTE2, id_l1, 0);      // WAR: wait for MTE1 to finish reading L1
TLOAD(aMatTile[id_l1], ...);
TLOAD(bMatTile[id_l1], ...);
rls_buf(PIPE_MTE2, id_l1, 0);      // RAW: signal L1 written

// MTE1: L1 → L0A/B
get_buf(PIPE_MTE1, id_l1, 0);      // RAW: wait for MTE2 to write L1
get_buf(PIPE_MTE1, id_l0, 0);      // WAR: wait for CUBE to finish reading L0
TMOV(aTile[id_l1], aMatTile[id_l1]);
TMOV(bTile[id_l1], bMatTile[id_l1]);
rls_buf(PIPE_MTE1, id_l1, 0);      // WAR: signal L1 free for MTE2 reuse
rls_buf(PIPE_MTE1, id_l0, 0);      // RAW: signal L0A/B written

// CUBE: TMATMUL / TMATMUL_ACC
get_buf(PIPE_M, id_l0,    0);      // RAW: wait for MTE1 to write L0A/B  ← ORDER MATTERS
get_buf(PIPE_M, C_BUF_ID, 0);      // WAR: wait for FIXPIPE to finish L0C ← must be AFTER id_l0
TMATMUL_ACC(cTile, cTile, aTile[id_l1], bTile[id_l1]);
rls_buf(PIPE_M, id_l0,    0);      // WAR: signal L0A/B free for MTE1 reuse
rls_buf(PIPE_M, C_BUF_ID, 0);      // RAW: signal L0C updated

// After K-loop — FIXPIPE: L0C → GM
get_buf(PIPE_FIX, C_BUF_ID, 0);   // RAW: wait for CUBE final result in L0C
TSTORE(dstGlobal, cTile);          // AccTile store → FIXPIPE (NOT MTE3)
rls_buf(PIPE_FIX, C_BUF_ID, 0);
```

> **Important:** Wrap all `get_buf`/`rls_buf` in `#ifndef __PTO_AUTO__` guards for auto-scheduling compatibility.

---

## Debug Journey: How We Fixed get_buf/rls_buf

### Root cause 1: Same buffer IDs for L1 and L0 tiles
L1 tiles and L0 tiles initially shared IDs 0–3. Using the same ID for two different memory levels at different pipeline stages causes spurious WAR/RAW conflicts. **Fix:** offset L0 IDs by `NUM_BUFS` (IDs 4–7).

### Root cause 2: Wrong pipe for L0C store guard
`TSTORE` of an `AccTile` is dispatched by **FIXPIPE** (`PIPE_FIX`), not MTE3. Using `PIPE_MTE3` as the guard is silently ignored — the store runs without waiting for CUBE to finish. **Fix:** use `PIPE_FIX`.

### How we found it
1. CPU sim passed (stubs are no-ops — no actual sync enforced)
2. A5 sim failed: max diff ~12.6, ~50% bad elements, normal cycle count (not a hang)
3. Inspected `core0.cubecore0.instr_log.dump`:
   - Saw `GET_BUF PIPE:CUBE bufId:8` for future iterations firing at tick ~1255
   - But `RLS_BUF PIPE:MTE2 bufId:0` (iter 0 L1 ready) only at tick ~1847
   - Confirmed CUBE was consuming stale L0 data from future iterations
4. Fixed pipe ordering and replaced `PIPE_MTE3` → `PIPE_FIX` for TSTORE guard
5. A5 sim: ✅ PASS, max diff 9.54e-06, bad count 0

### Diagnostic log
See `DEBUG.md` in this directory for the full instruction log analysis.

---

## Build & Run

```bash
# Set environment
source /usr/local/Ascend/cann/set_env.sh

# Build A5 sim
cd ~/pto-isa-ptoas-st
python3 tests/script/build_st.py -r sim -v a5 -t cube_matmul_4buf

# Generate test data (if not present)
python3 tests/cpu/st/testcase/cube_matmul_4buf/gen_data.py
cp tests/cpu/st/testcase/cube_matmul_4buf/CubeMatmul4BufTest.case_f16_32x1024_1024x256/*.bin \
   tests/npu/a5/src/st/build/bin/CubeMatmul4BufTest.case_f16_32x1024_1024x256/

# Run A5 sim
cd tests/npu/a5/src/st/build/bin
LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH \
  ./cube_matmul_4buf

# Build & run CPU sim (all variants)
python3 tests/run_cpu.py --testcase cube_matmul_4buf --clean --verbose
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `cube_matmul_4buf_kernel.cpp` | ✅ Main kernel — `get_buf`/`rls_buf` sync (v3, correct) |
| `cube_matmul_4buf_preload_kernel.cpp` | ✅ `set_flag`/`wait_flag` sync baseline |
| `cube_matmul_4buf_kernel_barrier.cpp` | ✅ `pipe_barrier(PIPE_ALL)` — simple but slow |
| `gen_data.py` | Test data generator (A, B matrices + golden C) |
| `main.cpp` | GTest harness (CPU sim) |
| `CMakeLists.txt` | Build config |
| `DEBUG.md` | Full debug log and instruction trace analysis |
| `README.md` | This file |

---

## References

- `npu_skills/pypto/bufid-cube-sync.md` — Complete get_buf/rls_buf reference for cube pipeline
- `npu_skills/debugging/cube-bufid-mismatch.md` — Debug playbook for this class of failure
- `npu_skills/pypto/ptoas-insert-sync-analysis.md` Section 10 — Buffer-ID mechanism overview
- CCE intrinsics: `__builtin_cce_get_buf`, `__builtin_cce_rls_buf`
