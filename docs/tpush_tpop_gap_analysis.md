# TPUSH/TPOP ISA Design — Comprehensive Gap Analysis

This document consolidates gaps in the current `pto_tpush_tpop_isa_design.md` specification based on:
1. **FA Performance Kernel** (`fa_performance_kernel.cpp`) — real production implementation
2. **ExpandMixedKernel Pass** (`pypto_expand_mixed_kernel_pass.md`) — compiler pass design

---

## Executive Summary: 7 Major Gaps

| # | Gap | Current Spec | Real-World Requirement |
|---|-----|--------------|------------------------|
| **1** | Subtile Tiling | 1:1 slot=tile | Slot >> AccTile >> VecTile with inner loops |
| **2** | GM vs Local Buffer | Fixed per platform | Compiler cost model decides |
| **3** | A5 Dual-Dst 1:2 | Not mentioned | Hardware row-split to both Vec cores |
| **4** | Intra-Core Sync | Cross-core only | Must unify FIXP→M, MTE→V with cross-core |
| **5** | Deferred Sync | Always immediate | Need SYNC_DEFERRED mode for subtile loops |
| **6** | Split vs Predication | Not addressed | Tensors may be unsplittable; need AIV_IDX predication |
| **7** | SPMD Integration | Not mentioned | spmd_idx/spmd_size interact with tpush/tpop |

---

## Gap 1: Subtile Tiling — Slot >> Tile

### Current Spec Assumption

The spec assumes **1:1 correspondence** between FIFO slot and local tile:
```
SLOT_SIZE = one tile
tpush moves one tile, tpop receives one tile
cross-core sync fires once per tile
```

### FA Kernel Reality

GM FIFO slot is **much larger** than local compute tiles:

```
FA Example (A3):
┌─────────────────────────────────────────────────────────────────┐
│  GM FIFO Slot:     128 × 256  (CUBE_S0 × TILE_S1)              │
│  Cube AccTile:     128 × 128  (CUBE_S0 × CUBE_S1)              │
│  Vec Tile:           8 × 256  (Vec_S0 × Tile_S1)               │
│                                                                 │
│  kTileFactor = TILE_S1 / CUBE_S1 = 256 / 128 = 2               │
│  VEC_CORES = 2                                                  │
│  Vec_S0 = CUBE_S0 / VEC_CORES / kTileFactor = 128/2/2 = 32     │
│    → per row_slice: Vec_S0 = 32 / kTileFactor = 8              │
└─────────────────────────────────────────────────────────────────┘
```

**Why different tiling?**

| Core | Tiling Axis | Reason |
|------|-------------|--------|
| **Cube** | Large M×N (128×128) | L0 buffer reuse, compute intensity |
| **Vector** | Large S1 (256) | Row-reduce along S1, amortize overhead |
| **Vec rows** | Small (8) | Fit in UB, kTileFactor row slices |

**Subtile loop structure:**

```cpp
// Cube: 2 sub_tile_id iterations per FIFO slot
for (int sub_tile_id = 0; sub_tile_id < kTileFactor; ++sub_tile_id) {
    // compute 128×128 AccTile, accumulate into 128×256 GM slot
}

// Vector: kTileFactor row_slice iterations per FIFO slot  
for (int row_slice = 0; row_slice < kTileFactor; ++row_slice) {
    // load 8×256 VecTile from GM slot, compute softmax
}
```

**Cross-core sync fires on LAST subtile only:**
```cpp
if (sub_tile_id == kTileFactor - 1)
    qk2smSync.record();  // signal full 128×256 slot ready
```

### Required Spec Extension

```cpp
// Subtile offset parameters:
tpush_to_aiv_subtile(TILE, AIV_IDX, row_offset, col_offset, SYNC_MODE)
tpop_from_aic_subtile(TILE, AIV_IDX, row_offset, col_offset)

// SYNC_MODE for deferred sync (see Gap 5):
enum SyncMode { SYNC_IMMEDIATE, SYNC_DEFERRED };
```

---

## Gap 2: GM vs Local Buffer — Compiler Cost Model

### Current Spec Assumption

Fixed mapping per platform:
- A3: always GM
- A5: always consumer's local SRAM

### Real-World Requirement

On A5, using local buffer is **not always optimal**. The ExpandMixedKernel pass or a cost model pass should decide:

| Factor | GM FIFO | Local Buffer |
|--------|---------|--------------|
| **Latency** | Higher | Lower |
| **Bandwidth** | DDR/HBM limited | SRAM higher |
| **Capacity** | GB | 512KB UB / L1 |
| **Zero-copy** | No | Yes |
| **Multi-consumer** | Easy | Hard (duplicate) |

**When GM might be better on A5:**
- Slot size exceeds local buffer capacity
- Multiple consumers need same data (broadcast)
- Ping-pong depth requires more slots than local buffer can hold

### Required Spec Extension

```cpp
enum BufferPlacement {
    PLACEMENT_GM,
    PLACEMENT_CONSUMER_UB,
    PLACEMENT_CONSUMER_L1,
    PLACEMENT_AUTO,  // compiler cost model decides
};

aic_initialize_pipe(DIR_MASK, SLOT_SIZE, PLACEMENT, ...)
```

---

## Gap 3: A5 Dual-Dst 1:2 Feature

### Current Spec Assumption

Not mentioned. Each `tpush_to_aiv` targets a single AIV_IDX.

### FA Kernel Reality

On A5, to address **QK compute → softmax memory bound**, hardware supports writing AccTile to **two destinations** simultaneously:

```
A5 Dual-Dst (1:2 ratio):

Cube AccTile [128, 128]
       │
       ├──────────────────► Vec0 UB [64, 128]  (upper half)
       │
       └──────────────────► Vec1 UB [64, 128]  (lower half)

The hardware infers this from the shape ratio:
  AccTile rows : VecTile rows × VEC_CORES = 128 : 64 × 2 = 1:2
```

### Required Spec Extension

```cpp
// Single push to both Vector cores with automatic row split:
tpush_to_aiv_dual(TILE)  // A5 only

// Shape requirement: TILE.rows == 2 × VecTile.rows × VEC_CORES
// No explicit AIV_IDX needed — hardware splits automatically
```

---

## Gap 4: Intra-Core Sync Unification

### Current Spec Assumption

Only covers **cross-core** sync (Cube ↔ Vector via flags). Does not address:
- Intra-core dependencies for local AccTile/VecTile buffers
- How FIXP→M, MTE→V interacts with TPUSH/TPOP

### FA Kernel Reality

Multiple layers of synchronization:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sync Hierarchy                               │
├─────────────────────────────────────────────────────────────────┤
│  1. INTRA-CORE (local buffer dependencies):                    │
│     Cube:  FIXP ──set_flag──► M ──wait_flag──► MTE             │
│     Vec:   MTE ──set_flag──► V ──wait_flag──► MTE3             │
│                                                                  │
│  2. CROSS-CORE (FIFO slot sync):                               │
│     qk2smSync: Cube ──record──► Vec                             │
│     sm2pvSync: Vec ──record──► Cube                             │
└─────────────────────────────────────────────────────────────────┘
```

**A5 with local buffer mapping changes the picture:**
- **Reverse dependency** (consumer signals "buffer free") becomes **inter-core**
- Buffer lives in consumer's SRAM, so "free" signal must cross cores

### Required Spec Extension

```cpp
// Before tpush: ensure local compute is done
tpush_wait_local(TILE, PIPE_FLAG)

// After tpop: signal local compute can proceed
tpop_signal_local(TILE, PIPE_FLAG)

// Combined operations:
tpush_to_aiv_with_local_wait(TILE, AIV_IDX, LOCAL_FLAG)
tpop_from_aic_with_local_signal(TILE, AIV_IDX, LOCAL_FLAG)
```

---

## Gap 5: Deferred Sync (SYNC_MODE)

### Current Spec Assumption

Every `tpush` implicitly does `SET flag_ready[tag]` after DMA. Sync is **immediate**.

### Subtile Loop Requirement

When Cube writes multiple subtiles into one GM slot, you want:
- Subtile 0: DMA, **no signal** (slot not yet complete)
- Subtile 1: DMA, **signal** (slot now complete)

This requires **deferred sync**:

```cpp
// Subtile 0: DMA only, no flag SET
tpush_to_aiv_subtile(tile_0, AIV_IDX, 0, 0, SYNC_DEFERRED)

// Subtile 1: DMA + flag SET
tpush_to_aiv_subtile(tile_1, AIV_IDX, 0, 64, SYNC_IMMEDIATE)
// OR: deferred DMA + explicit signal
tpush_to_aiv_subtile(tile_1, AIV_IDX, 0, 64, SYNC_DEFERRED)
tpush_signal(DIR_C2V, AIV_IDX)
```

### Required Spec Extension

```cpp
enum SyncMode {
    SYNC_IMMEDIATE,  // SET flag_ready after MTE (default)
    SYNC_DEFERRED,   // MTE only, no flag SET
};

// Explicit signal for deferred sync:
tpush_signal(DIR, AIV_IDX)  // SET flag_ready[tag]
```

---

## Gap 6: Split vs Predication (AIV_IDX Handling)

### Current Spec Assumption

`AIV_IDX` is passed to every tpush/tpop. Assumes all tensors are split evenly between the two Vector cores.

### ExpandMixedKernel Pass Reality

**Not all tensors can be split.** The pass performs axis analysis:

| Operation | Reduction Axis | Splittable Axes |
|---|---|---|
| `pl.sub/mul/div/exp` | None | Any |
| `pl.row_max/row_sum` | Last axis | All **except** last |
| `tensor.view/read/assemble` | None | Any |

**Unsplittable case:**
- `pl.row_sum` on 1D tensor `[N]` — only axis is reduction axis
- Cannot split; must **predicate** to `AIV_IDX==0` only

```python
# Unsplittable — predicated:
if AIV_IDX == 0:
    total = pl.row_sum(vec)  # full [512], only AIV_IDX==0

# If result needed by both cores: broadcast
```

**Impact on TPUSH/TPOP:**

| Case | AIV Kernel | AIC Kernel |
|------|------------|------------|
| **Split** | Both cores do tpush/tpop with half data | AIC does 2× tpush/tpop (half_0, half_1) |
| **Predicated** | Only AIV_IDX==0 does tpush/tpop | AIC does 1× tpush/tpop to AIV_IDX=0 |

### Required Spec Extension

The spec should explicitly document:

1. **Split mode** (default):
   ```cpp
   // AIV: both cores participate with half data
   tpush_to_aic(half_tile, AIV_IDX)
   
   // AIC: receive from both
   tpop_from_aiv(half_0, 0)
   tpop_from_aiv(half_1, 1)
   full = concat(half_0, half_1)
   ```

2. **Predicated mode** (for unsplittable tensors):
   ```cpp
   // AIV: only AIV_IDX==0 participates
   if (AIV_IDX == 0) {
       tpush_to_aic(full_tile, 0)
   }
   
   // AIC: receive from single core
   tpop_from_aiv(full_tile, 0)
   ```

3. **Broadcast mode** (predicated result needed by both AIV):
   ```cpp
   // After predicated computation, broadcast result
   if (AIV_IDX == 0) {
       tpush_to_aiv_peer(result, 1)  // push to AIV_IDX=1
   } else {
       result = tpop_from_aiv_peer(0)  // pop from AIV_IDX=0
   }
   ```

---

## Gap 7: SPMD Integration

### Current Spec Assumption

The spec covers single-cluster tpush/tpop only. No mention of how SPMD (multi-cluster) affects the ring buffer.

### ExpandMixedKernel Pass Reality

The pass introduces `call_spmd_function` and `call_spmd_group` for multi-cluster execution:

```python
# Launch the function group on 4 clusters in parallel
pl.call_spmd_group(paged_attention_group, args=(...), spmd_size=4)

# Expands to:
# Cluster 0: AIC(spmd_idx=0) + AIV(spmd_idx=0, AIV_IDX=0) + AIV(spmd_idx=0, AIV_IDX=1)
# Cluster 1: AIC(spmd_idx=1) + ...
# ...
```

**Each cluster is independent** — no cross-cluster tpush/tpop. But the spec should clarify:

1. **GM FIFO partitioning**: Each cluster uses a different `GM_SLOT_BUFFER` region
   ```cpp
   // Per-cluster GM buffer offset:
   GM_SLOT_BUFFER_CLUSTER = GM_SLOT_BUFFER_BASE + spmd_idx * SLOT_NUM * SLOT_SIZE
   ```

2. **Flag isolation**: Cross-core flags are cluster-local (already true by hardware)

3. **Initialization with spmd_idx**:
   ```cpp
   aic_initialize_pipe(..., spmd_idx, spmd_size)
   // Uses spmd_idx to compute cluster-specific buffer address
   ```

### Required Spec Extension

```cpp
// Extended initialization API for SPMD:
aic_initialize_pipe(DIR_MASK, SLOT_SIZE, 
                    GM_SLOT_BUFFER_BASE,  // global base (shared)
                    C2V_CONSUMER_BUF, V2C_CONSUMER_BUF,
                    spmd_idx, spmd_size)  // NEW: for GM partitioning

// Compiler computes per-cluster GM offset:
// GM_SLOT_BUFFER = GM_SLOT_BUFFER_BASE + spmd_idx * SLOT_NUM * SLOT_SIZE
```

---

## Updated Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Complete TPUSH/TPOP Architecture (v2)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                   GM FIFO SLOT (one per cluster)                   │      │
│  │                     [SlotRows × SlotCols]                          │      │
│  │  ┌────────────┬────────────┬────────────┬────────────┐            │      │
│  │  │ subtile 0  │ subtile 1  │ subtile 2  │ subtile 3  │ ← Cube     │      │
│  │  │ [R,C/k]    │ [R,C/k]    │ [R,C/k]    │ [R,C/k]    │   writes   │      │
│  │  │ DEFERRED   │ DEFERRED   │ DEFERRED   │ IMMEDIATE  │   k times  │      │
│  │  └────────────┴────────────┴────────────┴────────────┘            │      │
│  │  ┌─────────────────────────────────────────────────────┐          │      │
│  │  │ Vec0 rows [0, R/2)          Vec1 rows [R/2, R)      │ ← Vec    │      │
│  │  │  OR: Predicated (AIV_IDX==0 only)                   │   reads  │      │
│  │  └─────────────────────────────────────────────────────┘          │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  Control Plane (unified intra + cross-core):                                │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                                                                    │      │
│  │   Cube                           Vec0                  Vec1       │      │
│  │     │                              │                     │        │      │
│  │   FIXP→M (local)                 MTE→V (local)        MTE→V      │      │
│  │     │                              │                     │        │      │
│  │     ├─── slot_ready (IMMEDIATE) ──►├◄── slot_ready ─────┤        │      │
│  │     │    or tpush_signal()         │                     │        │      │
│  │     │                              │                     │        │      │
│  │     │◄── slot_free ────────────────┤───── slot_free ────►│        │      │
│  │     │    (split: 2 signals)        │    (split: 2 signals)       │      │
│  │     │    (predicated: 1 signal)    │    (predicated: 1)          │      │
│  │                                                                    │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  SPMD (multi-cluster):                                                      │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                                                                    │      │
│  │   Cluster 0        Cluster 1        Cluster 2        Cluster 3   │      │
│  │   spmd_idx=0       spmd_idx=1       spmd_idx=2       spmd_idx=3  │      │
│  │   ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐ │      │
│  │   │ GM FIFO │      │ GM FIFO │      │ GM FIFO │      │ GM FIFO │ │      │
│  │   │ [i*off] │      │ [i*off] │      │ [i*off] │      │ [i*off] │ │      │
│  │   └────┬────┘      └────┬────┘      └────┬────┘      └────┬────┘ │      │
│  │        │                │                │                │      │      │
│  │    AIC+AIV×2        AIC+AIV×2        AIC+AIV×2        AIC+AIV×2  │      │
│  │   (independent)    (independent)    (independent)    (independent)      │
│  │                                                                    │      │
│  │   No cross-cluster tpush/tpop — each cluster is self-contained   │      │
│  │                                                                    │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Spec Update Checklist

### API Extensions

```cpp
// 1. Subtile access with offset and sync mode
tpush_to_aiv_subtile(TILE, AIV_IDX, row_off, col_off, SYNC_MODE)
tpop_from_aic_subtile(TILE, AIV_IDX, row_off, col_off)

// 2. Explicit signal for deferred sync
tpush_signal(DIR, AIV_IDX)

// 3. A5 dual-dst (automatic row split to both Vec cores)
tpush_to_aiv_dual(TILE)

// 4. Local dependency integration
tpush_wait_local(TILE, PIPE_FLAG)
tpop_signal_local(TILE, PIPE_FLAG)

// 5. Buffer placement enum
enum BufferPlacement { PLACEMENT_GM, PLACEMENT_CONSUMER_UB, PLACEMENT_CONSUMER_L1, PLACEMENT_AUTO };

// 6. Vec-to-Vec peer communication (for broadcast after predication)
tpush_to_aiv_peer(TILE, PEER_AIV_IDX)
tpop_from_aiv_peer(PEER_AIV_IDX)

// 7. SPMD-aware initialization
aic_initialize_pipe(..., spmd_idx, spmd_size)
aiv_initialize_pipe(..., spmd_idx, spmd_size)
```

### Documentation Additions

1. **Subtile Tiling Section**: Explain SLOT >> TILE pattern, inner loops, deferred sync
2. **Split vs Predication Section**: Document when tensors can/cannot be split
3. **A5 Dual-Dst Section**: Document 1:2 ratio inference and automatic row split
4. **Intra-Core Sync Section**: Explain FIXP→M, MTE→V integration
5. **SPMD Section**: Document multi-cluster GM partitioning and cluster independence

---

## Cross-Reference to ExpandMixedKernel Pass

The ExpandMixedKernel pass (`pypto_expand_mixed_kernel_pass.md`) is the compiler component that:

1. **Colors IR nodes** (WHITE/GREEN/RED) to identify AIC vs AIV operations
2. **Inserts tpush/tpop** at cross-color boundaries
3. **Performs axis analysis** to determine split vs predication
4. **Emits InCoreFunctionGroup** with co-scheduled AIC + AIV kernels
5. **Handles SPMD** via `call_spmd_group` expansion

The TPUSH/TPOP spec should reference this pass as the primary consumer of the ISA, and the pass documentation should reference the spec for instruction semantics.

---

## References

- `kernels/manual/a2a3/flash_atten/fa_performance_kernel.cpp` — production FA kernel
- `pypto/docs/pypto_expand_mixed_kernel_pass.md` — compiler pass design
- `include/pto/npu/a2a3/custom/TSync_Custom.hpp` — sync primitives
- `include/pto/npu/a2a3/custom/TSyncCVID.hpp` — CV sync API
