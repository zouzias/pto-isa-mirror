# MLA DN Performance Optimization - Debug Log

## Objective
Fix V reconstruction correctness bug in MLA DN kernel (FIFO_MODE=1 ALL_UB path) on A5 NPU Ascend simulator, then verify performance gain vs baseline.

## Test Configuration
- **Command**: `source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh && cd kernels/manual/a5/MLA && bash run.sh -r sim -v Ascend950PR_9599 --cases "128,256,128,512,128,128" -p 2 -m 1`
- **Case**: HEAD=128, LATENT=256, S0=128, S1=512
- **CUBE_S0=128, CUBE_S1=128, TILE_S1=128** (kTileFactor=1, num_tiles_s1=4)
- **FIFO_MODE=1** (ALL_UB path), Ascend950PR_9599 simulator
- **Baseline** (without V recon): max diff ~0.015

## Key Tile Dimensions & Buffer Counts
- V recon matmul: `pto_macro_matmul<Cube_S1=128, Cube_LATENT=256, Cube_HEAD=128>(cKv, w_uv, vReconsAccTile, Init)` — NT layout, **kSegments=2** (fittingCubeK=128, Tile_K=256)
- PV matmul: `pto_macro_matmul<Cube_S0=128, Cube_S1=128, Cube_HEAD=128>(pMat, vMat, pvAccTile, pvAccMode)` — TN layout, kSegments=1
- cKvMatTNBuffers = (qkPreloadNum+1)*kTileFactor = 3*1 = 3
- w_uvMatTNBuffers = 1
- vMatTNBuffers = 2
- pMatTNBuffers = 3
- vReconsAccTile shares pvAccTile's L0C via TASSIGN (same address)

## L0C Buffer Layout (A5 = 256KB)
- Two halves: 0x0 (128KB) and 0x20000 (128KB)
- qkAccTile, pvAccTile, vReconsAccTile all share same L0C base address
- TASSIGN routes TMATMUL output to 0x0 or 0x20000 based on evtID ping-pong

## assign_running_acc_tile evtID Trace (SEPARATED counters)

**After fix**: QK and PV use separate template-tagged counters (QKAccTag / PVAccTag).

- QKAccTag init: assign_running_acc_tile<QKAccTag>(qkAccTile, 0) → idx=0, TASSIGN(0x0), idx→1
- PVAccTag init: assign_running_acc_tile<PVAccTag>(pvAccTile, 1) → idx=1, TASSIGN(0x20000), idx→0

**QK evtID pattern**: 0, 1, 0, 1, 0, 1, ... (standard alternation after preload)
**PV evtID pattern**: 1, 0, 1, 0, 1, 0, ... (starts at 1, alternates)

QK and PV always use OPPOSITE L0C halves. No L0C collision.

## Experiments & Findings (Sessions 1-2)

### Experiment 1: V Recon Correctness Verification
**Method**: TSTORE dump of vReconsAccTile to pv_tile_fifo
**Result**: V recon data **CORRECT for ALL 4 tiles** — max diff 0.000002 vs expected V = c_kv @ W_uv
**Conclusion**: V recon matmul and TASSIGN produce correct data. Bug is downstream.

### Experiment 2: PV Result Verification
**Method**: TSTORE dump of pvAccTile to pv_tile_fifo
**Result**: PV tiles **0,1 are ALL ZEROS**, tiles 2,3 have wrong values
**Conclusion**: PV pipeline fails for first 2 tiles when V recon is present.

### Experiment 3: L0C Address Fix (0x10000 → 0x20000)
**Result**: Did NOT fix PV zeros issue. Address fix was correct for A5 but not root cause.

### Experiment 4: Opposite L0C Address for vReconsAccTile
**Result**: PV tile 0 still zeros. Opposite halves don't work — must share same half.

### Experiment 5: No V Recon (Comment Out Entire Section)
**Result**: ALL PV tiles are ALL ZEROS (vMatTile has stale data → P × 0 = 0)
**Conclusion**: PV pipeline runs but produces zeros without V data.

### Experiment 6: vMatTile Dump Attempt
**Result**: Build error — TSTORE only supports Vec/Acc TileType, not Mat.
**Conclusion**: Cannot directly dump vMatTile. Need indirect verification.

### Experiment 7: Separate Static Counter for QK and PV
**Method**: Template-tagged assign_running_acc_tile<QKAccTag/PVAccTag> with separate static counters
**Result**: PV evtID pattern now 1,0,1,0 (standard alternation). QK evtID: 0,1,0,1.
**Test**: S1=512 with -m 1 → **FAILED** — max diff 1.57, PV tiles 0,1 still ALL ZEROS
**Conclusion**: Counter separation alone doesn't fix the bug.

### Experiment 8: Remove Debug TSTOREs
**Method**: Removed both TSTORE vReconsAccTile and TSTORE pvAccTile debug dumps
**Result**: Same failure pattern persists (PV tiles 0,1 zeros)
**Conclusion**: Debug TSTOREs not the root cause (Candidate E ruled out).

---

## Session 3: PIPE_FIX→PIPE_MTE1 Sync Fix

### Root Cause Analysis

The V recon pipeline flow is:
1. V recon TMATMUL → vReconsAccTile (PIPE_M writes to L0C)
2. TMOV vReconsAccTile → vMatTile (PIPE_FIX writes L0C data to L1)
3. PV TMATMUL's TEXTRACT reads vMatTile from L1 (PIPE_MTE1 reads L1→L0A/L0B)

**The problem**: There is no synchronization between PIPE_FIX (which writes V to L1) and PIPE_MTE1 (which reads V from L1). TEXTRACT can read stale L1 data before TMOV completes writing.

In the reference flash_atten_mxfp8 kernel, V comes from GM TLOAD (PIPE_MTE2→PIPE_MTE1), which has built-in GM DMA synchronization. But in MLA with V recon, V comes from TMOV (PIPE_FIX→L1), which needs explicit PIPE_FIX→PIPE_MTE1 sync.

### Fix Applied: PIPE_FIX→PIPE_MTE1 Event Sync

Added two sync operations in compute_pv:

1. **After V recon TMOV** (line 534): `set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(svMatTileEventId));`
   - PIPE_FIX signals PIPE_MTE1 that V L1 write is complete after TMOV

2. **Before PV TMATMUL** (line 565): `wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(svMatTileEventId));`
   - PIPE_MTE1 waits for V L1 data before TEXTRACT

The `svMatTileEventId` parameter (previously unused in compute_pv) alternates between PV_EVENT_ID0=2 and PV_EVENT_ID1=3 (ping-pong based on `pv_src_pingpong_id % vMatTNBuffers + PV_EVENT_ID0`). This allows overlapping iterations on different event IDs.

### Experiment 9: PIPE_FIX→PIPE_MTE1 Sync WITHOUT Priming
**Method**: Added set_flag/wait_flag PIPE_FIX→PIPE_MTE1 with svMatTileEventId, no init-phase priming
**S1=512 Result**: **PASS** — max diff 0.000571 ✓
**S1=1024 Result**: **FAIL** — max diff 0.519872, 98.8% elements wrong
**S1=4096 Result**: **FAIL** — max diff 0.387582, 98.9% elements wrong
**Conclusion**: PIPE_FIX→PIPE_MTE1 sync works for 4 tiles but breaks for 8+ tiles.

### Experiment 10: PIPE_FIX→PIPE_MTE1 Sync WITH Priming
**Method**: Added init-phase priming: `set_flag(PIPE_FIX, PIPE_MTE1, PV_EVENT_ID0)` and `set_flag(PIPE_FIX, PIPE_MTE1, PV_EVENT_ID1)` plus teardown waits
**S1=1024 Result**: **FAIL** — max diff 0.661615, plus simulator errors: `execute_set_flag already has same set_flag!`
**Conclusion**: Priming causes `execute_set_flag already has same set_flag` errors on the Ascend simulator. This confirms that the Ascend simulator uses **binary flags** (not queued events) for set_flag/wait_flag. When a flag is already set (from the priming or a previous iteration), a second set_flag on the same (srcPipe, dstPipe, eventId) channel triggers an error.

### Critical Insight: Binary Flag Event Model on Ascend Simulator

The Ascend simulator error `execute_set_flag already has same set_flag!` confirms:
- **Event channels are binary flags** (set/clear, not queues/ counters)
- A second `set_flag` on an already-set flag channel is an **error**
- This means PIPE_FIX→PIPE_MTE1 with ping-pong event IDs (PV_EVENT_ID0=2, PV_EVENT_ID1=3) cannot work across multiple iterations because:
  - Iter 0: set_flag(PIPE_FIX, PIPE_MTE1, 2) sets flag 2
  - Iter 0: wait_flag(PIPE_FIX, PIPE_MTE1, 2) clears flag 2 ← OK within same iteration
  - BUT: if PIPE_FIX processes iter 2's set_flag(PIPE_FIX, PIPE_MTE1, 2) while PIPE_MTE1 hasn't consumed iter 0's set_flag(2) yet → flag 2 is already set → simulator error!

The set_flag/wait_flag pair within one iteration SHOULD work (set after TMOV, wait before TEXTRACT). But on the NPU simulator, the CCE compiler compiles instructions into pipe-specific queues that process concurrently. PIPE_FIX might advance to iter 2's set_flag(2) before PIPE_MTE1 has consumed iter 0's wait_flag(2). This causes the "already has same set_flag" error.

**Why S1=512 (4 tiles) passes**: With only 4 tiles, the pipeline queues are short enough that PIPE_FIX and PIPE_MTE1 stay roughly synchronized. PIPE_FIX doesn't advance far enough to trigger event ID collision with a future iteration.

**Why S1=1024+ fails**: With 8+ tiles, PIPE_FIX processes instructions faster than PIPE_MTE1, causing iter N+2's set_flag to collide with iter N's unconsumed set_flag on the same event channel.

---

## Root Cause: PIPE_FIX→PIPE_MTE1 Binary Flag Collision Across Iterations

The fundamental issue is that **Ascend uses binary flags per (srcPipe, dstPipe, eventId) channel**, not queued events. This means:

1. A set_flag can only be issued when the flag is CLEAR (not already set)
2. A wait_flag clears the flag after consuming it
3. Two set_flags on the same channel must have a wait_flag between them

For the V recon TMOV→PV TEXTRACT sync:
- Each iteration's set_flag/wait_flag pair uses the same channel (PIPE_FIX→PIPE_MTE1, svMatTileEventId)
- svMatTileEventId alternates between 2 and 3 (2 channels)
- But with pipeline overlap across iterations, PIPE_FIX can issue iter N+2's set_flag(2) while iter N's set_flag(2) hasn't been consumed yet
- The simulator rejects the second set_flag because flag 2 is already set

---

## Remaining Candidates & Next Steps

### Candidate A (CONFIRMED): Pipeline Event Collision on Binary Flag Channels
PIPE_FIX→PIPE_MTE1 sync cannot use the same event ID across multiple iterations because Ascend binary flags don't allow two consecutive set_flags without a wait_flag between them.

**Fix options**:

### Option 1: GM Roundtrip for V (TSTORE L0C→GM + TLOAD GM→L1)
- Replace TMOV vReconsAccTile→vMatTile with: TSTORE vReconsAccTile→GM + TLOAD vMatTile from GM
- V data goes through GM, using the standard PIPE_MTE2→PIPE_MTE1 TLOAD path
- The PIPE_MTE1→PIPE_MTE2 sync (svMatTileEventId) with proper priming works for V TLOAD (same pattern as flash_atten)
- Requires: dedicated GM buffer for V, __gm__ half *v_data parameter, TSTORE/TLOAD instead of TMOV
- **Pros**: Uses proven pipeline pattern from flash_atten reference kernel
- **Cons**: GM roundtrip adds latency (2 GM accesses per tile), some performance loss vs direct TMOV
- **Implementation**: Already partially designed (see Priority 1 in old next steps)

### Option 2: More Event IDs for PIPE_FIX→PIPE_MTE1
- Use 3 or 4 event IDs instead of 2 (e.g., PV_EVENT_ID0=2, PV_EVENT_ID1=3, PV_EVENT_ID2=4)
- svMatTileEventId = pv_src_pingpong_id % N + PV_EVENT_ID0, where N >= 3
- More event IDs means more spacing between iterations that reuse the same channel
- With N=3: iter 0 uses channel 2, iter 1 uses 3, iter 2 uses 4, iter 3 uses 2 again (3-tile spacing)
- **Pros**: No GM roundtrip, maintains TMOV L0C→L1 direct path
- **Cons**: May still fail for very long sequences (S1=4096=32 tiles); needs enough event IDs (Ascend supports 8 per pipe pair)
- **Risk**: If PIPE_FIX processes 3+ iterations before PIPE_MTE1 catches up, still collides

### Option 3: PIPE_MTE1→PIPE_FIX Reverse Sync (Pipeline Release Pattern)
- After PV TMATMUL's TEXTRACT reads V from L1, signal PIPE_FIX that L1 V buffer is free
- `set_flag(PIPE_MTE1, PIPE_FIX, <eventId>)` after TEXTRACT
- `wait_flag(PIPE_MTE1, PIPE_FIX, <eventId>)` before V recon TMOV in next iteration
- This creates a proper producer-consumer pipeline with priming (similar to flash_atten's V TLOAD pipeline)
- **Pros**: Clean pipeline design with proper priming, proven pattern
- **Cons**: Requires careful event ID management; need to add PIPE_MTE1→PIPE_FIX priming in init phase

### Option 4: Disable V Recon, Use GM V Load (Baseline Comparison)
- Temporarily disable V recon and add GM V load path for performance comparison
- This gives us the baseline performance numbers to compare against V recon
- **Purpose**: Establish performance target before optimizing V recon pipeline

### Recommended Next Steps (Priority Order)

1. **Try Option 2 first** (more event IDs): Quick change to svMatTileEventId formula, test S1=1024/4096
   - Change: `svMatTileEventId = pv_src_pingpong_id % 3 + PV_EVENT_ID0` (use 3 IDs instead of 2)
   - Add PV_EVENT_ID2=4 to CoreEvtID enum
   - Add priming for all 3 channels in init phase
   - Add teardown waits for all 3 channels
   - Test S1=512, S1=1024, S1=4096

2. **If Option 2 fails, try Option 1** (GM roundtrip):
   - Add `__gm__ half *v_data` parameter to runTMLA
   - Replace TMOV with TSTORE+TLOAD in compute_pv
   - Use PIPE_MTE1→PIPE_MTE2 sync (svMatTileEventId) with priming (same as flash_atten)
   - Test correctness first, then measure performance vs baseline

3. **If Option 1 works, implement Option 3** (PIPE_MTE1→PIPE_FIX pipeline) for optimal performance:
   - This eliminates the GM roundtrip while maintaining correct pipeline behavior
   - More complex to implement but potentially best performance

---

## Current Code State (After Session 3)
- V recon section present (TASSIGN, TMATMUL, PIPE_M→PIPE_FIX sync, TMOV)
- PIPE_FIX→PIPE_MTE1 set_flag/wait_flag added (with svMatTileEventId)
- Init-phase priming for PIPE_FIX→PIPE_MTE1 (PV_EVENT_ID0, PV_EVENT_ID1) — CAUSES ERRORS, should be removed
- Teardown waits for PIPE_FIX→PIPE_MTE1 — should be removed with priming
- Debug TSTOREs removed (both vReconsAccTile and pvAccTile)
- Separate QKAccTag/PVAccTag counters in assign_running_acc_tile
- L0C address fix (0x20000) in assign_running_acc_tile
- **S1=512 passes, S1=1024+ fails due to binary flag collision**

## Files Modified
- `kernels/manual/a5/MLA/mla_performance_dn_kernel.cpp`:
  - Line 534: `set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(svMatTileEventId));` (after V recon TMOV)
  - Line 565: `wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(svMatTileEventId));` (before PV TMATMUL)
  - Lines 1120-1121: Init priming `set_flag(PIPE_FIX, PIPE_MTE1, PV_EVENT_ID0/1)` — **NEEDS REMOVAL** (causes errors)
  - Lines 1280-1281: Teardown `wait_flag(PIPE_FIX, PIPE_MTE1, PV_EVENT_ID0/1)` — **NEEDS REMOVAL**
  - Lines 332-333: QKAccTag/PVAccTag structs for separate counters
  - Lines 1021-1023: Init with separate counters (QKAccTag initial=0, PVAccTag initial=1)

## Reference Kernel
- `/home/jyc304691735/Projects/jinlin/pto-isa/kernels/manual/a5/flash_atten_mxfp8/fa_performance_dn_kernel.cpp`
  - Uses PIPE_MTE1→PIPE_MTE2 sync (svMatTileEventId) for V TLOAD pipeline
  - Has priming: `set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0/1/2/3)` in init phase
  - Has teardown: `wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0/1/2)` at end
  - V comes from GM TLOAD, not from L0C→L1 TMOV
  - Pipeline works for all S1 sizes (proven on both simulator and real A5)

---

## Session 4: GM Roundtrip for V Reconstruction (PIPE_M→PIPE_MTE2 Approach)

### Approach
Replace TMOV (L0C→L1, PIPE_FIX domain) with TSTORE (L0C→GM, PIPE_FIX) + TLOAD (GM→L1, PIPE_MTE2).
This avoids PIPE_FIX→PIPE_MTE1 binary flag collision by routing V data through GM instead of direct L1 write.

### Key Design Decisions
- **TSTORE float→half conversion**: L0C TileAcc stores float data; TSTORE to `GlobalTensor<half>` automatically applies F322F16 quantization mode (float→half conversion on PIPE_FIX). No explicit quantization parameter needed. Verified in PTO library TStore.hpp: `GetCastPreQuantMode<float, half>() = F322F16`.
- **TLOAD half→half**: TLOAD requires same sizeof on both sides (`sizeof(Tile::DType) == sizeof(GlobalTensor::DType)` enforced). `GlobalTensor<half>` → `Tile<TileType::Mat, half>` works (ND2NZ layout conversion).
- **PIPE_M→PIPE_MTE2 sync**: PIPE_M signals PIPE_MTE2 that V recon data is in GM after PIPE_FIX frees L0C. PIPE_M is the bottleneck (slow TMATMUL), so it can't get ahead of PIPE_MTE2 — no binary flag collision risk.
- **PIPE_MTE2→PIPE_MTE1 forward sync**: PIPE_MTE2 signals PIPE_MTE1 that V data is in L1 (replaces old PIPE_FIX→PIPE_MTE1).
- **PIPE_MTE1→PIPE_MTE2 reverse sync**: PIPE_MTE1 signals PIPE_MTE2 that V L1 buffer is consumed (replaces old PIPE_MTE1→PIPE_FIX reverse sync).

### Sync Flow (New Approach)

1. V recon TMATMUL → vReconsAccTile in L0C (PIPE_M)
2. PIPE_M→PIPE_FIX: set_flag(EVENT_ID2) — TMATMUL completion
3. PIPE_FIX: wait_flag(EVENT_ID2), TSTORE(vReconsAccTile → GM), set_flag(PIPE_FIX→PIPE_M, accTileEvtID) — free L0C
4. P data loading (unchanged)
5. PIPE_M: wait_flag(PIPE_FIX→PIPE_M, accTileEvtID) — L0C free, then set_flag(PIPE_M→PIPE_MTE2, vReconsGMEventId) — signal GM data ready
6. PIPE_MTE2: wait_flag(PIPE_M→PIPE_MTE2, vReconsGMEventId), wait_flag(PIPE_MTE1→PIPE_MTE2, svMatTileEventId) — reverse sync
7. PIPE_MTE2: TLOAD(vMatTile from GM), set_flag(PIPE_MTE2→PIPE_MTE1, svMatTileEventId) — V data ready
8. PIPE_MTE1: wait_flag(PIPE_MTE2→PIPE_MTE1, svMatTileEventId) — V data ready confirmed
9. PV TMATMUL (PIPE_M + PIPE_MTE1)
10. PIPE_MTE1: set_flag(PIPE_MTE1→PIPE_MTE2, svMatTileEventId) — reverse sync: V L1 consumed

### vReconsGMEventId
- Alternating: `(tile_id * kTileFactor + sub_tile_id) % 2` → EVENT_ID0/1 on PIPE_M→PIPE_MTE2 pair
- No priming needed: PIPE_M naturally sets the flag after receiving L0C free signal. PIPE_MTE2 waits (blocks) until signal arrives.

### Priming Changes
- **Removed**: PIPE_MTE1→PIPE_FIX priming (old reverse sync, caused collision)
- **Added**: PIPE_MTE1→PIPE_MTE2 priming (PV_EVENT_ID0/1) — for reverse sync wait before TLOAD
- **NOT added**: PIPE_M→PIPE_MTE2 priming — would cause collision with first compute_pv set_flag

### GM Buffer
- `__gm__ half *v_recons_fifo`: Per-block FIFO in GM, size = block_rows * qkp_tile_fifo_size * kTileFactor * Cube_S1 * HEAD_SIZE * sizeof(half)
- Per-sub_tile offset: `(tile_id * kTileFactor + sub_tile_id) * Cube_S1 * HEAD_SIZE`
- GlobalTensor layout: `GlobalTensor<half, Shape<1,1,1,Cube_S1,HEAD_SIZE>, Stride<1,1,1,HEAD_SIZE,1>>` (ND row-major)

### Experiment 11: First GM Roundtrip Attempt (WITH PIPE_M→PIPE_MTE2 Priming)
**Method**: Full GM roundtrip implementation with PIPE_M→PIPE_MTE2 priming (set_flag EVENT_ID0/1 in init phase)
**S1=512 Result**: **FAIL** — simulator errors: `execute_set_flag already has same set_flag!` (4 collisions)
- Output: ALL ZEROS (max diff 1.187, 99.9% elements wrong)
**Root Cause**: PIPE_M→PIPE_MTE2 priming sets EVENT_ID0/1 in init phase. First compute_pv iteration tries set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0) for vReconsGMEventId=0 — but EVENT_ID0 is already set from priming! Binary flag collision.

### Experiment 12: Remove PIPE_M→PIPE_MTE2 Priming — S1=512 (partial success)
**Method**: Removed PIPE_M→PIPE_MTE2 priming (set_flag/wait_flag EVENT_ID0/1) from init/teardown
**S1=512 Result**: **PASS** (max diff 0.000571) but still 4 `execute_set_flag already has same set_flag!` errors
**Root Cause**: The 4 collision errors came from PIPE_MTE1→PIPE_MTE2 event ID overlap between QK and PV:
- QK uses EVENT_ID0/1/2 (values 0,1,2) for QK reverse sync priming
- PV uses PV_EVENT_ID0/1 (values 2,3) for PV reverse sync priming
- EVENT_ID2 (value 2) = PV_EVENT_ID0 (value 2) — **same channel on PIPE_MTE1→PIPE_MTE2!**
- Setting the same event channel twice in init phase = binary flag collision
- Additionally, QK compute uses qkMatTileEventId cycling through 0,1,2; PV uses svMatTileEventId cycling through 2,3
- Pipeline overlap can cause QK set_flag(2) colliding with PV set_flag(2) on the same channel

### Experiment 13: Fix Event ID Overlap (QK=0,1,2 / PV=3,4)
**Method**: Changed CoreEvtID enum: QK_EVENT_ID0=0, QK_EVENT_ID1=1, QK_EVENT_ID2=2, PV_EVENT_ID0=3, PV_EVENT_ID1=4
- Updated init priming to use named constants: `set_flag(PIPE_MTE1, PIPE_MTE2, QK_EVENT_ID0/1/2)` and `set_flag(PIPE_MTE1, PIPE_MTE2, PV_EVENT_ID0/1)`
- Updated teardown waits similarly
- No overlap: QK events 0-2, PV events 3-4, PIPE_M→PIPE_MTE2 events 0-1 (different pipe pair)
**S1=512 Result**: **PASS** — max diff 0.000571, **NO collision errors** ✓
**S1=1024 Result**: **PASS** — max diff 0.000150, **NO collision errors** ✓
**S1=4096 Result**: **PASS** — max diff 0.000199, **NO collision errors** ✓

### Final Code State (After Session 4 — CORRECTNESS VERIFIED)
- V recon TMOV replaced with TSTORE (float L0C → half GM, F322F16 auto conversion)
- V recon L1 loading: TLOAD (half GM → half L1 Mat tile, ND2NZ)
- PIPE_M→PIPE_MTE2 sync (vReconsGMEventId alternating EVENT_ID0/1, no priming needed)
- PIPE_MTE2→PIPE_MTE1 forward sync (svMatTileEventId PV_EVENT_ID0/1)
- PIPE_MTE1→PIPE_MTE2 reverse sync (svMatTileEventId PV_EVENT_ID0/1, with priming)
- CoreEvtID: QK_EVENT_ID0=0, QK_EVENT_ID1=1, QK_EVENT_ID2=2, PV_EVENT_ID0=3, PV_EVENT_ID1=4 (no overlap)
- PIPE_MTE1→PIPE_MTE2 priming: QK_EVENT_ID0/1/2 (QK reverse sync) + PV_EVENT_ID0/1 (PV reverse sync)
- PIPE_MTE1→PIPE_FIX priming REMOVED
- PIPE_FIX→PIPE_M priming kept (EVENT_ID0/1)
- v_recons_fifo GM buffer added throughout (kernel, host, header, macro)
- **CORRECTNESS VERIFIED for S1=512, 1024, 4096**

### Next Steps (Performance)
1. Measure performance (tick count) with V recon vs baseline (V from GM without V recon)
2. If performance regression from GM roundtrip, consider Option 3 (PIPE_MTE1→PIPE_FIX pipeline with proper spacing)
3. Consider using L0C→L1 direct path with more event IDs (Option 2) if GM roundtrip latency is too high
