# TPUSH/TPOP ISA vs ExpandMixedKernel Pass — Interface Gap Analysis

This document identifies **interface gaps** between what the ExpandMixedKernel compiler pass needs and what the current TPUSH/TPOP ISA spec provides.

---

## Interface Requirements from ExpandMixedKernel Pass

The pass generates the following IR nodes that must map to ISA instructions:

| Pass Output (IR) | ISA Instruction Needed |
|------------------|------------------------|
| `aic_initialize_pipe(...)` | ✅ Defined |
| `aiv_initialize_pipe(...)` | ✅ Defined |
| `tpush_to_aiv(tile, AIV_IDX)` | ✅ Defined |
| `tpush_to_aic(tile, AIV_IDX)` | ✅ Defined |
| `tpop_from_aic(tile, AIV_IDX)` | ✅ Defined |
| `tpop_from_aiv(tile, AIV_IDX)` | ✅ Defined |
| `pl.reserve_buffer(...)` | ✅ Defined |
| `pl.import_peer_buffer(...)` | ✅ Defined |
| `call_group @group(...)` | ❌ **NOT in ISA** (codegen concern) |
| `InCoreFunctionGroup` | ❌ **NOT in ISA** (IR-level concept) |

---

## Gap 8: Tile Shape/Size in TPUSH/TPOP Interface

### Pass Requirement

The pass generates tpush/tpop with **tile variables** that have shape information:

```python
# Pass output:
tpush_to_aiv(qi_half, AIV_IDX)  # qi_half has shape [8, 128], dtype=float16

tpop_from_aic(sij_half, AIV_IDX)  # sij_half has shape [8, 256], dtype=float32
```

### ISA Gap

The current spec only mentions `SLOT_SIZE` (bytes) at initialization time. It does not specify:

1. **Per-operation tile shape**: Does each tpush/tpop carry shape info, or is it inferred from the tile variable?
2. **Dynamic shapes**: Can tile sizes vary between tpush/tpop calls in the same kernel?
3. **Shape mismatch handling**: What happens if producer pushes [16, 128] but consumer pops [8, 256]?

### Required Spec Clarification

```cpp
// Option A: Shape in tile metadata (implicit)
// Tile IR node carries shape, tpush/tpop just reference it
tpush_to_aiv(tile, AIV_IDX)  // shape from tile.shape

// Option B: Explicit shape parameters
tpush_to_aiv(tile, AIV_IDX, rows, cols, dtype)

// Option C: Fixed slot size, tile must fit
// SLOT_SIZE >= tile.size_bytes(), shape irrelevant to ISA
```

**Recommendation**: Option A (shape in tile metadata) aligns with the pass design. The ISA should clarify that:
- `SLOT_SIZE` at init = maximum tile size in bytes
- Each tpush/tpop transfers `tile.size_bytes()` ≤ `SLOT_SIZE`
- Shape information flows through tile metadata, not ISA instructions

---

## Gap 9: Multiple Tile Types in Same Pipe

### Pass Requirement

The pass generates multiple tpush/tpop pairs for **different tiles** with different shapes:

```python
# AIC Kernel:
tpop_from_aiv(qi_half, 0)   # [8, 128] fp16
tpop_from_aiv(kj_half, 0)   # [64, 128] fp16
# ...
tpush_to_aiv(sij_half, 0)   # [64, 256] fp32
# ...
tpop_from_aiv(pij_half, 0)  # [64, 256] fp16
tpop_from_aiv(vj_half, 0)   # [64, 128] fp16
```

### ISA Gap

The current spec assumes a single `SLOT_SIZE` per direction. It does not address:

1. **Variable-sized tiles**: Can different tpush/tpop calls use different slot sizes?
2. **Multiple logical pipes**: Should there be one pipe per tile type, or one shared pipe?
3. **Tag/FIFO interleaving**: How does the round-robin tag work with mixed tile sizes?

### Required Spec Clarification

**Option A: Single pipe, max slot size**
```cpp
// At init: SLOT_SIZE = max of all tile sizes
aic_initialize_pipe(DIR_MASK, MAX_SLOT_SIZE, ...)

// Each tpush/tpop uses its actual tile size (≤ MAX_SLOT_SIZE)
// Same tag sequence for all tiles — FIFO order preserved
tpush_to_aiv(qi, 0)    // uses tag 0, actual bytes = qi.size
tpush_to_aiv(kj, 0)    // uses tag 1, actual bytes = kj.size
```

**Option B: Multiple logical pipes**
```cpp
// Separate pipes for different communication patterns
aic_initialize_pipe(PIPE_QK, ..., SLOT_SIZE_QK, ...)
aic_initialize_pipe(PIPE_PV, ..., SLOT_SIZE_PV, ...)

tpush_to_aiv(qi, 0, PIPE_QK)
tpush_to_aiv(pij, 0, PIPE_PV)
```

**Recommendation**: Option A (single pipe, max slot) is simpler and matches the FA kernel pattern. The ISA should clarify that:
- `SLOT_SIZE` = maximum tile size across all tpush/tpop in the kernel
- Tag sequence is shared across all tile types (FIFO ordering)
- Actual transfer size is per-tile, not per-slot

---

## Gap 10: Concat/Split Operations

### Pass Requirement

The pass generates `concat` and `first_half`/`second_half` operations:

```python
# AIC kernel receives two halves:
tpop_from_aiv(qi_half_0, 0)
tpop_from_aiv(qi_half_1, 1)
qi = concat(qi_half_0, qi_half_1)  # Combine into full tile

# AIC kernel sends two halves:
tpush_to_aiv(first_half(sij), 0)
tpush_to_aiv(second_half(sij), 1)
```

### ISA Gap

The ISA does not specify:

1. **Concat/split primitives**: Are these separate operations, or inlined into tpush/tpop?
2. **Memory layout**: Where are `half_0` and `half_1` stored before concat?
3. **Zero-copy concat**: On A5, can concat be avoided if both halves are contiguous?

### Required Spec Clarification

```cpp
// Option A: Separate buffer + explicit concat (current assumption)
LocalTensor<half, [8, 128]> qi_half_0, qi_half_1;
tpop_from_aiv(qi_half_0, 0);  // DMA to qi_half_0
tpop_from_aiv(qi_half_1, 1);  // DMA to qi_half_1
qi = concat(qi_half_0, qi_half_1);  // CPU/MTE copy

// Option B: Direct pop into full tile with offset
LocalTensor<half, [16, 128]> qi;
tpop_from_aiv_at(qi, 0, row_offset=0);   // DMA to qi[0:8, :]
tpop_from_aiv_at(qi, 1, row_offset=8);   // DMA to qi[8:16, :]
// No concat needed — data lands in final position

// Option C: Scatter/gather DMA (hardware feature)
tpop_from_aiv_gather(qi, {AIV_0: [0:8, :], AIV_1: [8:16, :]});
```

**Recommendation**: Option B (`tpop_from_aiv_at` with offset) avoids extra copy. The ISA should add:

```cpp
// Pop into existing tile at offset:
tpop_from_aiv_at(TILE, AIV_IDX, row_offset, col_offset)

// Push from tile at offset:
tpush_to_aiv_from(TILE, AIV_IDX, row_offset, col_offset, rows, cols)
```

---

## Gap 11: DIR_MASK Determination

### Pass Requirement

The pass must determine `DIR_MASK` from the cross-color data flow:

```python
# Pass analysis:
for each cross-color tile T:
    if T flows RED→GREEN: DIR_C2V needed
    if T flows GREEN→RED: DIR_V2C needed

DIR_MASK = computed_directions
```

### ISA Gap

The current spec mentions `DIR_C2V` and `DIR_V2C` but does not specify:

1. **Bidirectional support**: Can `DIR_MASK = DIR_C2V | DIR_V2C`?
2. **Slot allocation**: How are slots divided for bidirectional?
3. **Tag interleaving**: Do C2V and V2C share the same tag sequence?

### Required Spec Clarification

```cpp
// Bidirectional example:
#define DIR_C2V  0x01
#define DIR_V2C  0x02
#define DIR_BOTH (DIR_C2V | DIR_V2C)

// Slot division for bidirectional (SLOT_NUM=8):
// C2V: slots 0-3 (tags 0-3)
// V2C: slots 4-7 (tags 0-3 in V2C space)

// Alternative: shared slots, interleaved tags
// tag 0 = C2V[0], tag 1 = V2C[0], tag 2 = C2V[1], ...
```

The spec already mentions "SLOT_NUM=4 per direction" for bidirectional. This should be clarified with explicit examples.

---

## Gap 12: Error Handling / Deadlock Prevention

### Pass Requirement

The pass generates correct push/pop pairing. But what if:
- Producer pushes more than consumer pops (overflow)
- Consumer pops more than producer pushes (underflow)
- Push/pop order mismatch between AIC and AIV

### ISA Gap

The spec does not address:

1. **Overflow detection**: What happens if producer fills all slots and pushes again?
2. **Underflow detection**: What happens if consumer pops from empty?
3. **Timeout**: Is there a way to detect deadlock at runtime?
4. **Debug aids**: Can flag state be read for debugging?

### Required Spec Clarification

```cpp
// Overflow: producer blocks indefinitely (WAIT flag_free never signals)
// Underflow: consumer blocks indefinitely (WAIT flag_ready never signals)

// Optional debug API:
uint32_t tpush_try(TILE, AIV_IDX, timeout_cycles);  // returns 0=success, 1=timeout
uint32_t tpop_try(TILE, AIV_IDX, timeout_cycles);

// Flag state query (debug only):
uint8_t get_flag_state(DIR, AIV_IDX, tag);  // 0=free, 1=ready
```

---

## Gap 13: AIV_IDX Implicit vs Explicit

### Pass Requirement

The pass parameterizes the AIV kernel with `AIV_IDX`:

```python
def mixed_kernel_aiv(..., AIV_IDX):
    tpush_to_aic(qi_half, AIV_IDX)  # AIV_IDX from parameter
```

### ISA Gap

The current spec treats `AIV_IDX` as an explicit parameter to each instruction. But:

1. **Implicit AIV_IDX**: On some platforms, the Vector core knows its own index. Should AIV_IDX be implicit?
2. **get_aiv_idx() intrinsic**: Is there a way to query the current core's AIV_IDX?

### Required Spec Clarification

```cpp
// Option A: Explicit parameter (current spec)
tpush_to_aic(tile, AIV_IDX)  // caller passes AIV_IDX

// Option B: Implicit from hardware
tpush_to_aic(tile)  // uses get_aiv_idx() internally

// Option C: Both supported
tpush_to_aic(tile)           // implicit AIV_IDX
tpush_to_aic(tile, explicit) // explicit override for cross-core
```

**Recommendation**: Provide `get_aiv_idx()` intrinsic; allow both implicit and explicit modes.

```cpp
// Intrinsic to query current AIV core index (Vector kernel only):
uint8_t get_aiv_idx();  // returns 0 or 1

// Implicit form (common case):
tpush_to_aic(tile)  // equivalent to tpush_to_aic(tile, get_aiv_idx())
```

---

## Gap 14: Finalization / Teardown

### Pass Requirement

The pass generates initialization but not finalization. After the kernel completes:

1. Are there pending flags that need cleanup?
2. What happens if kernel exits with unreleased slots?

### ISA Gap

The spec only covers `initialize_pipe`. No mention of:

1. **finalize_pipe()**: Cleanup at kernel end?
2. **Implicit cleanup**: Hardware auto-resets on kernel exit?
3. **Flag state persistence**: Do flags persist across kernel launches?

### Required Spec Clarification

```cpp
// Option A: Explicit finalize (symmetric with initialize)
aic_finalize_pipe();

// Option B: Implicit cleanup on kernel exit
// Hardware resets all flags and ring buffer state

// Option C: Flag state persists, caller must drain
// (Useful for pipelined kernel chains)
```

**Recommendation**: Option B (implicit cleanup) is simplest. Document that:
- All flags are reset to initial state (consumer pre-signals free)
- Ring buffer pointers are reset
- No explicit finalize needed

---

## Summary: Interface Gaps

| Gap # | Category | Status |
|-------|----------|--------|
| 8 | Tile shape in interface | Need clarification |
| 9 | Multiple tile types | Need clarification |
| 10 | Concat/split operations | Need `tpop_at`/`tpush_from` |
| 11 | DIR_MASK bidirectional | Partially covered, need examples |
| 12 | Error/deadlock handling | Need debug aids |
| 13 | AIV_IDX implicit/explicit | Need `get_aiv_idx()` |
| 14 | Finalization | Need clarification (implicit cleanup) |

---

## Appendix: Complete API Surface (Proposed)

```cpp
// ==================== Initialization ====================
void aic_initialize_pipe(
    uint8_t  DIR_MASK,           // DIR_C2V | DIR_V2C
    uint32_t SLOT_SIZE,          // max tile size in bytes
    __gm__ void* GM_SLOT_BUFFER, // A3: required; A5: optional
    void*    C2V_CONSUMER_BUF,   // A5: consumer UB address (from import)
    void*    V2C_CONSUMER_BUF,   // A5: consumer L1 address (from import)
    uint32_t spmd_idx,           // SPMD: cluster index (for GM partitioning)
    uint32_t spmd_size           // SPMD: total clusters
);

void aiv_initialize_pipe(
    uint8_t  DIR_MASK,
    uint32_t SLOT_SIZE,
    __gm__ void* GM_SLOT_BUFFER,
    void*    C2V_CONSUMER_BUF,
    void*    V2C_CONSUMER_BUF,
    uint32_t spmd_idx,
    uint32_t spmd_size
);

// ==================== Data Transfer ====================
// Basic (full tile):
void tpush_to_aiv(TILE, uint8_t AIV_IDX);
void tpush_to_aic(TILE, uint8_t AIV_IDX);
void tpop_from_aic(TILE, uint8_t AIV_IDX);
void tpop_from_aiv(TILE, uint8_t AIV_IDX);

// With offset (subtile / scatter-gather):
void tpush_to_aiv_from(TILE, AIV_IDX, row_off, col_off, rows, cols, SYNC_MODE);
void tpop_from_aic_at(TILE, AIV_IDX, row_off, col_off);

// Deferred sync:
void tpush_signal(uint8_t DIR, uint8_t AIV_IDX);

// A5 dual-dst (both AIV cores):
void tpush_to_aiv_dual(TILE);

// ==================== Local Sync Integration ====================
void tpush_wait_local(TILE, PIPE_FLAG);
void tpop_signal_local(TILE, PIPE_FLAG);

// ==================== Vec-to-Vec (Broadcast after Predication) ====================
void tpush_to_aiv_peer(TILE, uint8_t PEER_AIV_IDX);
TILE tpop_from_aiv_peer(uint8_t PEER_AIV_IDX);

// ==================== Intrinsics ====================
uint8_t get_aiv_idx();   // Current Vector core index (0 or 1)
uint32_t get_spmd_idx(); // Current cluster index

// ==================== Debug (Optional) ====================
uint32_t tpush_try(TILE, AIV_IDX, timeout_cycles);
uint32_t tpop_try(TILE, AIV_IDX, timeout_cycles);
uint8_t get_flag_state(DIR, AIV_IDX, tag);

// ==================== Buffer Reservation (DSL) ====================
// Consumer side:
buffer = pl.reserve_buffer(name, size, base=pl.AUTO);

// Producer side:
peer_buf = pl.import_peer_buffer(name, peer_func);
```

---

## References

- `pypto/docs/pypto_expand_mixed_kernel_pass.md` — ExpandMixedKernel pass design
- `docs/pto_tpush_tpop_isa_design.md` — current ISA spec
- `kernels/manual/a2a3/flash_atten/fa_performance_kernel.cpp` — production FA kernel
