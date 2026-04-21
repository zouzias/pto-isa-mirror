# cube_matmul_4buf Buffer-ID Debug

## Context

**Test case:** `tests/npu/a5/src/st/testcase/cube_matmul_4buf/`
**Branch:** `ptoas-small-tile-st`
**Repo:** `https://gitcode.com/ChanKaLok/pto-isa`
**Server:** `ssh happybot@192.168.0.106`
**Build dir:** `~/pto-isa-ptoas-st`

## Problem Statement

Three sync methods were tested on A5 simulator:

| Sync Method | Kernel File | Cycles | Max Diff | Status |
|-------------|-------------|--------|----------|--------|
| `set_flag`/`wait_flag` | `cube_matmul_4buf_kernel_barrier.cpp` | 24,276 | 9.54e-06 | ✅ PASSED |
| `pipe_barrier(PIPE_ALL)` | barrier variant | 52,957 | 9.54e-06 | ✅ PASSED (2x slower) |
| `get_buf`/`rls_buf` | `cube_matmul_4buf_kernel.cpp` | 23,803 | **12.6** | ❌ FAILED |

**Bad count:** 8031 / 8192 elements wrong. First elements show garbage/uninitialized values.
**Suspicious:** Buffer-ID version has slightly *fewer* cycles than set_flag — suggests sync is not blocking at all.

## Buffer-ID Correctness Principles (from RAG)

- `get_buf<PIPE_X>(id)` blocks until previous pipeline called `rls_buf<PIPE_X>(id)` OR another pipeline released it for X to consume
- Handles **both** RAW (read-after-write) and WAR (write-after-read) implicitly with `mode=0`
- No priming or draining needed
- **Same ID can be reused across different pipeline stages for the same physical buffer**
- **Different buffers at different memory levels (L1 vs L0A vs L0B vs L0C) must use DIFFERENT buffer IDs** — IDs are tied to physical resource identity

## Lok's Key Clarification

> "same atile and btile matile pair can use same id across different pipelines but matile and left/right tile can work in different level pipeline (GM->L1->L0) need (MTE2->MTE1 and MTE1->CUBE, CUBE->FIXPIPE) different sync so better use different buffer id"

This means:
- A_L1 and A_L0 (aTile, aMatTile) are at **different memory levels** → need **different IDs**
- B_L1 and B_L0 (bTile, bMatTile) are at **different memory levels** → need **different IDs**
- cTile (L0C) needs its own ID
- buf_id 0-3 should NOT be reused for both L1 and L0 buffers

## Current Kernel's Buffer-ID Assignment

```cpp
constexpr int C_BUF_ID = 8;

// In loop: buf_id = k % 4 (0-3)
get_buffer<PIPE_MTE2>(buf_id);   // GM→L1: aMatTile0/bMatTile0 (L1)
TLOAD(aMatTile, ...); TLOAD(bMatTile, ...);
rls_buffer<PIPE_MTE2>(buf_id);

get_buffer<PIPE_MTE1>(buf_id);   // ← SAME buf_id for L0 as L1!
TMOV(aTile, aMatTile); TMOV(bTile, bMatTile);
rls_buffer<PIPE_MTE1>(buf_id);

get_buffer<PIPE_M>(buf_id);      // ← SAME buf_id again for CUBE!
get_buffer<PIPE_M>(C_BUF_ID);
TMATMUL / TMATMUL_ACC;
rls_buffer<PIPE_M>(buf_id);
rls_buffer<PIPE_M>(C_BUF_ID);
```

## Suspected Bug: ID Collision Between Memory Levels

The same `buf_id` (0-3) is used for:
1. L1 tiles (aMatTile, bMatTile) — managed by MTE2/MTE1
2. L0 tiles (aTile, bTile) — managed by MTE1/PIPE_M

Since these are physically different buffers at different memory levels, they should have separate IDs. Using the same ID likely creates incorrect dependency tracking in hardware.

**Proposed fix:** Use separate ID ranges:
- `buf_id_l1 = k % 4` (IDs 0-3) for L1 A/B tiles
- `buf_id_l0 = k % 4 + 4` (IDs 4-7) for L0A/L0B tiles
- `C_BUF_ID = 8` for L0C (keep as-is)

## Debug Plan

### Step 1: Reproduce the failure
```bash
ssh happybot@192.168.0.106
source /usr/local/Ascend/cann/set_env.sh
cd ~/pto-isa-ptoas-st
python3 tests/script/build_st.py -r sim -v a5 -t cube_matmul_4buf
cd tests/npu/a5/src/st/build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib
./bin/cube_matmul_4buf
```

### Step 2: Compare buffer dump logs (L0A/L0B/L0C/L1)
The A5 simulator produces buffer dump logs. Compare the passing (set_flag) vs failing (get_buf) kernel dumps:
- Check L0A dump: is A data correct before TMATMUL?
- Check L0B dump: is B data correct before TMATMUL?
- Check L0C dump: is accumulator correct after TMATMUL?
- Find the **first iteration** where data diverges

### Step 3: Try fix with separate ID ranges for L1 vs L0
Create `cube_matmul_4buf_bufid_v2_kernel.cpp` with:
- L1 tiles: buf_id 0-3
- L0 tiles: buf_id 4-7
- C tile: buf_id 8

### Step 4: Verify mode parameter
Current: `get_buf(pipe, id, 0)` — mode=0
Check if mode should be non-zero for cube pipeline.

## Files

- **Failing kernel:** `tests/npu/a5/src/st/testcase/cube_matmul_4buf/cube_matmul_4buf_kernel.cpp`
- **Passing reference:** `tests/npu/a5/src/st/testcase/cube_matmul_4buf/cube_matmul_4buf_kernel_barrier.cpp` (set_flag version)
- **Test harness:** `tests/npu/a5/src/st/testcase/cube_matmul_4buf/main.cpp`
- **Build:** `python3 tests/script/build_st.py -r sim -v a5 -t cube_matmul_4buf`

## Notes

- The set_flag kernel was previously added to `main.cpp` as a separate test function — check if it can be toggled via `#define` or separate test case
- Need to enable buffer dump logging in simulator (check build script flags or env vars)
- Ivan may have debug skills for cube case — L0A/L0B format, NZ log in L1 log are key concepts per Xson

---

## A5 Sim Log Analysis (v2 kernel, 2026-04-21)

### CPU Sim: ✅ ALL 4 TESTS PASS
- `CubeMatmul4BufTest` (set_flag baseline): max diff 1.14e-05
- `CubeMatmul4BufPreloadTest`: max diff 1.14e-05
- `CubeMatmul4BufEventSyncTest`: max diff 1.14e-05
- `CubeMatmul4BufBufIdV2Test` (v2, separate L1/L0 IDs): max diff 1.14e-05

### A5 Sim: ❌ STILL FAILING (v2 kernel)
- max diff: 12.6, bad count: 8039 / 8192
- Cycles: 23,734 (same as broken v1)

### Root Cause (from instr_log.dump)

Inspecting `core0.cubecore0.instr_log.dump`:

```
[00001115] GET_BUF  PIPE:MTE2, bufId:0, cnt:0       ← iter 0 starts loading
[00001138] GET_BUF  PIPE:MTE2, bufId:1, cnt:0       ← iter 1 starts loading
[00001255] GET_BUF  PIPE:CUBE, bufId:8, cnt:1       ← iter 1's C acquire ALREADY QUEUED
[00001284] GET_BUF  PIPE:CUBE, bufId:8, cnt:2       ← iter 2's C acquire queued
[00001328] GET_BUF  PIPE:CUBE, bufId:8, cnt:3       ← iter 3's C acquire queued
...
[00001847] RLS_BUF  PIPE:MTE2, bufId:0, cnt:0       ← iter 0 L1 data ready (finally)
[00001848] GET_BUF  PIPE:MTE1, bufId:0, cnt:1
[00001849] GET_BUF  PIPE:MTE1, bufId:4, cnt:0
[00001916] RLS_BUF  PIPE:MTE1, bufId:0, cnt:1       ← iter 0 L0 data ready
[00001917] RLS_BUF  PIPE:MTE1, bufId:4, cnt:0
[00001918] GET_BUF  PIPE:CUBE, bufId:4, cnt:1       ← L0 slot acquire
[00001919] GET_BUF  PIPE:CUBE, bufId:8, cnt:0       ← iter 0 C acquire (finally!)
[00001977] CUBE MMAD  (iter 0 runs correctly)
[00001978] RLS_BUF  PIPE:CUBE, bufId:4, cnt:1
[00001979] RLS_BUF  PIPE:CUBE, bufId:8, cnt:0
```

**The problem:** C_BUF_ID=8 acquires for iter 1,2,3... (cnt:1,2,3...) are being issued at ticks
1255-1404 — BEFORE the MTE2 for iter 0 has even finished (tick 1847). The hardware is
pre-issuing future CUBE stage C-slot acquires out of order, ahead of the L0 data being ready.

The `GET_BUF PIPE:CUBE bufId:8 cnt:1` at tick 1255 doesn't stall because bufId:8 was just
released from a previous state (initial free), so it immediately proceeds — but the
corresponding L0 data (bufId:4..7) isn't loaded yet!

**Hypothesis:** The C_BUF_ID=8 acquire and the id_l0 acquire are done in program order
(get_buffer<PIPE_M>(id_l0) THEN get_buffer<PIPE_M>(C_BUF_ID)), but the hardware pipeline
is speculatively queuing C_BUF_ID=8 requests from future loop iterations out of order.

**This is not an ID collision bug** — it is a **pipeline reordering / speculative execution**
issue where get_buf for the C tile (single shared across all iters) can be satisfied
immediately in future iterations before the L0 data for those iterations is available.

### Next Steps
- [ ] Try ordering: acquire id_l0 FIRST, then C_BUF_ID. But since they're both PIPE_M, they
      should be in-order...
- [ ] Check if PIPE_M actually serializes both gets, or if only one is needed
- [ ] Try: single combined get_buf<PIPE_M>(id_l0) that covers both A/B and C
- [ ] Ask Ivan / check ISA docs: does PIPE_M pipeline allow multiple concurrent get_buf?
