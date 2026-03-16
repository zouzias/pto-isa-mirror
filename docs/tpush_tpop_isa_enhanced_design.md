# Enhanced TPUSH/TPOP ISA Design for Intra-Cluster Function Group Data Communication

## Overview

This document specifies an enhanced ISA design for `TPUSH` and `TPOP` instructions to support **intra-cluster data communication across InCore kernels** within a function group.

### Cluster Architecture

Each cluster contains **1 Cube core** and **2 buddy Vector cores** that share a hardware flag-based synchronization mechanism:

```
┌─────────────────────── Cluster ───────────────────────┐
│                                                       │
│  ┌──────────┐    flags (8 per dir)    ┌──────────┐   │
│  │  Vector 0 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    │   Cube   │   │
│                                       │          │   │
│  ┌──────────┐    flags (8 per dir)    │          │   │
│  │  Vector 1 │◄══════════════════════►│          │   │
│  └──────────┘   SET/WAIT V→C, C→V    └──────────┘   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### AIV Mode: 1:1 vs 1:2 Communication

The TPUSH/TPOP design supports two AIV communication modes:

| AIV_MODE | Description | Use Case |
|---|---|---|
| `AIV_SINGLE` (1:1) | Communication with **one** buddy Vector core only (AIV0 or AIV1) | Simple workloads, single Vector core active |
| `AIV_ALL` (1:2) | Communication with **both** buddy Vector cores simultaneously | Full cluster utilization, tile split across AIV0 and AIV1 |

```
AIV_SINGLE (1:1):

    Cube ◄────────────► Vector 0 only
         or
    Cube ◄────────────► Vector 1 only

AIV_ALL (1:2):

    Cube ◄────────────► Vector 0
         ◄────────────► Vector 1
         (same FIFO slot, tile split by cut-M or cut-N)
```

### Cut Axis for V2C Direction (AIV_ALL mode)

When using `AIV_ALL` mode in the **V2C direction** (Vector→Cube), the user must specify how the two AIV tiles are assembled into a single L1 MatTile:

| CUT_AXIS | Description | L1 Layout |
|---|---|---|
| `CUT_M` | Split along M (rows): AIV0 writes upper half, AIV1 writes lower half | `[N1][M1][M0][N0]` with AIV0→rows 0-7, AIV1→rows 8-15 |
| `CUT_N` | Split along N (cols): AIV0 writes left half, AIV1 writes right half | `[N1][M1][M0][N0]` with AIV0→cols 0-63, AIV1→cols 64-127 |

```
CUT_M (rows split):

    AIV0 UB [8, 128]  ──┐
                        ├──► L1 MatTile [16, 128]
    AIV1 UB [8, 128]  ──┘    (NZ layout, strided writes)

CUT_N (cols split):

    AIV0 UB [16, 64]  ──┐
                        ├──► L1 MatTile [16, 128]
    AIV1 UB [16, 64]  ──┘    (NZ layout, strided writes)
```

---

## API Definition

### New Constants

```cpp
enum AIVMode : uint8_t {
    AIV_SINGLE = 0,   // 1:1 mode: communicate with one AIV only
    AIV_ALL    = 1,   // 1:2 mode: communicate with both AIV0 and AIV1
};

enum CutAxis : uint8_t {
    CUT_NONE = 0,     // No cut (for AIV_SINGLE mode or C2V direction)
    CUT_M    = 1,     // Cut along M (rows)
    CUT_N    = 2,     // Cut along N (cols)
};
```

### Updated Initialization APIs

#### `aic_initialize_pipe(..., V2C_AIV_MODE, V2C_CUT_AXIS)`

**Called on the Cube (AIC) core at kernel startup.**

| Parameter | Type | Description |
|---|---|---|
| `DIR_MASK` | `uint8_t` | Bitmask of active directions (`DIR_C2V`, `DIR_V2C`, or both) |
| `SLOT_SIZE` | `uint32_t` | Size of each ring buffer slot in bytes |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM buffer (A2A3) or `nullptr` (A5) |
| `C2V_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base for C2V direction |
| `V2C_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base for V2C direction (Cube's L1) |
| **`V2C_AIV_MODE`** | `AIVMode` | **NEW**: `AIV_SINGLE` (1:1) or `AIV_ALL` (1:2) for V2C direction |
| **`V2C_CUT_AXIS`** | `CutAxis` | **NEW**: `CUT_M` or `CUT_N` when `V2C_AIV_MODE=AIV_ALL` |

**V2C Direction Handling**:
- `V2C_AIV_MODE=AIV_SINGLE`: Standard 1:1 communication. Each `tpop_from_aiv(AIV_IDX)` receives a full tile from the specified AIV.
- `V2C_AIV_MODE=AIV_ALL`: Both AIV0 and AIV1 push to the **same FIFO slot** with strided writes. The `V2C_CUT_AXIS` determines how tiles are assembled in L1.

#### `aiv_initialize_pipe(..., V2C_AIV_MODE, V2C_CUT_AXIS)`

**Called on a Vector (AIV) core at kernel startup.**

| Parameter | Type | Description |
|---|---|---|
| `DIR_MASK` | `uint8_t` | Bitmask of active directions |
| `SLOT_SIZE` | `uint32_t` | Size of each ring buffer slot in bytes |
| `GM_SLOT_BUFFER` | `__gm__ void*` | GM buffer (A2A3) or `nullptr` (A5) |
| `C2V_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base for C2V direction (Vector's UB) |
| `V2C_CONSUMER_BUF` | `uint32_t` | Consumer's SRAM base for V2C direction |
| **`V2C_AIV_MODE`** | `AIVMode` | **NEW**: `AIV_SINGLE` or `AIV_ALL` for V2C direction |
| **`V2C_CUT_AXIS`** | `CutAxis` | **NEW**: `CUT_M` or `CUT_N` when `V2C_AIV_MODE=AIV_ALL` |

**V2C Direction Handling (Vector as producer)**:
- `V2C_AIV_MODE=AIV_SINGLE`: This AIV pushes a full tile independently.
- `V2C_AIV_MODE=AIV_ALL`: This AIV computes its strided write offset based on `AIV_IDX` and `V2C_CUT_AXIS`, writing to the correct region of the shared L1 slot.

---

## Updated Instruction Behavior

### `tpush_to_aic(TILE, AIV_IDX)` — V2C Direction

**Executed on Vector (AIV).** Behavior depends on `V2C_AIV_MODE`:

**AIV_SINGLE mode**:
```
function tpush_to_aic(TILE, AIV_IDX):
    // Standard: full tile to dedicated slot
    WAIT flag_free[V2C, AIV_IDX]: target_tag
    MTE_copy(src=TILE, dst=slot[target_tag], size=SLOT_SIZE)
    SET flag_ready[V2C, AIV_IDX]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

**AIV_ALL mode**:
```
function tpush_to_aic(TILE, AIV_IDX):
    // Strided write: AIV0 and AIV1 write to SAME slot with offset
    WAIT flag_free[V2C, AIV_IDX]: target_tag
    
    // Compute strided write parameters based on CUT_AXIS and AIV_IDX
    offset = compute_aiv_offset(AIV_IDX, V2C_CUT_AXIS, TILE_SHAPE)
    stride = compute_dst_stride(V2C_CUT_AXIS, L1_LAYOUT)
    
    MTE_strided_copy(
        src=TILE,
        dst=slot[target_tag] + offset,
        src_stride=UB_STRIDE,
        dst_stride=stride
    )
    
    SET flag_ready[V2C, AIV_IDX]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

### `tpop_from_aiv(TILE, AIV_IDX)` — V2C Direction

**Executed on Cube (AIC).** Behavior depends on `V2C_AIV_MODE`:

**AIV_SINGLE mode**:
```
function tpop_from_aiv(TILE, AIV_IDX):
    // Standard: receive full tile from specified AIV
    WAIT flag_ready[V2C, AIV_IDX]: target_tag
    // On A5: zero-copy, TILE references L1 directly
    SET flag_free[V2C, AIV_IDX]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

**AIV_ALL mode**:
```
function tpop_from_aiv(TILE, AIV_IDX):
    // Wait for this AIV's data (both AIV0 and AIV1 must complete for full tile)
    WAIT flag_ready[V2C, AIV_IDX]: target_tag
    
    // On A5: zero-copy, data already in L1 via strided writes
    // TILE references the combined L1 slot
    
    SET flag_free[V2C, AIV_IDX]: target_tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

**Note**: In `AIV_ALL` mode, the consumer (Cube) must call `tpop_from_aiv` for **both** AIV0 and AIV1 to ensure both strided writes are complete before using the assembled tile.

---

## Example: V2C with AIV_ALL and CUT_M

```python
# Vector InCore (producer, runs on both AIV0 and AIV1)
@pl.incore
def vector_kernel(..., AIV_IDX):
    aiv_initialize_pipe(
        DIR_V2C,
        SLOT_SIZE=4096,
        ...,
        V2C_AIV_MODE=AIV_ALL,
        V2C_CUT_AXIS=CUT_M       # User specifies cut-M
    )
    
    for ...:
        tile = compute_half_tile()   # [8, 128] for this AIV
        tpush_to_aic(tile, AIV_IDX)  # Strided write to L1

# Cube InCore (consumer)
@pl.incore
def cube_kernel(...):
    aic_initialize_pipe(
        DIR_V2C,
        SLOT_SIZE=4096,
        ...,
        V2C_AIV_MODE=AIV_ALL,
        V2C_CUT_AXIS=CUT_M       # Must match Vector's setting
    )
    
    for ...:
        tpop_from_aiv(tile, 0)   # Wait for AIV0's strided write
        tpop_from_aiv(tile, 1)   # Wait for AIV1's strided write
        # Now tile is fully assembled [16, 128] in L1
        matmul(tile, ...)
```

---

## Summary of Changes

| Aspect | Original | Enhanced |
|---|---|---|
| **AIV Modes** | Implicit 1:1 | Explicit `AIV_SINGLE` (1:1) or `AIV_ALL` (1:2) |
| **Cut Axis** | Not specified | User specifies `CUT_M` or `CUT_N` during init |
| **V2C tpush** | Full tile to slot | Strided write when `AIV_ALL` |
| **V2C tpop** | Receive from one AIV | Must pop from both AIV0 and AIV1 when `AIV_ALL` |
| **FIFO Slot** | One slot per AIV | Shared slot for both AIVs when `AIV_ALL` |

The `AIV_MODE` and `CUT_AXIS` parameters are specified during `initialize_pipe`, enabling the compiler/runtime to:
1. Compute correct strided write offsets for each AIV
2. Allocate shared vs separate FIFO slots
3. Generate appropriate synchronization patterns
