# tpush_to_aic and tpop_from_aiv FIFO PTO-IR Design to PTO-ISA Issue Analysis: AIV→AIC Data Assembly Challenges

## Executive Summary

This document analyzes the V→C (Vector to Cube) data path challenges when lowering `tpush_to_aic` / `tpop_from_aiv` operations from PTO-IR to PTO-ISA. The core issue is:

1. **Two separate UB buffers** (AIV0 UB, AIV1 UB) must be assembled into a **single L1 MatTile** for CUBE consumption
2. **Performance goal**: UB→L1 strided write (fused TINSERT) for direct assembly during transfer
3. **Fallback only**: L1→L1 copy would be a fallback if we cannot infer how to lower separated `tpop` + `assemble` to a single strided `tpush`
4. The solution uses **FIFO definition** to convey cut-M/cut-N semantics, enabling direct strided writes

---

## 1. Background: V→C Data Path

### 1.1 Hardware Constraints

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI Core                                         │
│                                                                              │
│  ┌─────────────────┐                           ┌─────────────────┐          │
│  │    AIV0 UB      │                           │    AIV1 UB      │          │
│  │                 │                           │                 │          │
│  │  VecTile [8,128]│                           │  VecTile [8,128]│          │
│  │  (upper half M) │                           │  (lower half M) │          │
│  └────────┬────────┘                           └────────┬────────┘          │
│           │                                             │                    │
│           │ MTE2 (ub2l1)                                │ MTE2 (ub2l1)       │
│           │ copy_ubuf_to_cbuf                           │ copy_ubuf_to_cbuf  │
│           │ ⭐ STRIDED WRITE ⭐                          │ ⭐ STRIDED WRITE ⭐ │
│           │                                             │                    │
│           ▼                                             ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                            L1 Buffer                                 │    │
│  │                                                                      │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │              MatTile [16, 128] (NZ Layout)                  │    │    │
│  │   │              Physical: [8][1][16][16] in L1                 │    │    │
│  │   │                                                             │    │    │
│  │   │    ┌─────────────────┬─────────────────┐                   │    │    │
│  │   │    │  From AIV0 UB   │  From AIV1 UB   │                   │    │    │
│  │   │    │  rows 0-7       │  rows 8-15      │  (Cut-M)          │    │    │
│  │   │    └─────────────────┴─────────────────┘                   │    │    │
│  │   │                                                             │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   ❌ NO L1→L1 PATH: L1→L1 copy is fallback only, NOT performance    │    │
│  │   ✅ GOAL: Direct strided UB→L1 writes (fused TINSERT)              │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points**:
- **Performance Goal**: UB→L1 strided write that directly forms the assembled tile
- **L1→L1 is fallback only**: Would be used if we cannot infer how to fuse `tpop` + `assemble` into strided `tpush`
- No L1→L1 data path exists in current hardware anyway

### 1.2 NZ Layout Detail for [16, 128] Tile

For CUBE computation, activation tiles use **NZ (non-transposed) layout**:

```
Logical Tile: [M=16, N=128] (bf16/fp16)

Physical NZ Layout in L1: [N1][M1][M0][N0] = [8][1][16][16]

Where:
  N = N1 × N0 = 8 × 16 = 128
  M = M1 × M0 = 1 × 16 = 16

Memory Layout (8 blocks of 16×16):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Block 0 (n1=0)      Block 1 (n1=1)      ...      Block 7 (n1=7)            │
│  cols 0-15           cols 16-31                   cols 112-127              │
│  ┌──────────────┐    ┌──────────────┐            ┌──────────────┐           │
│  │ [M0=16][N0=16]│   │ [M0=16][N0=16]│           │ [M0=16][N0=16]│          │
│  │              │    │              │            │              │           │
│  │  row 0-15    │    │  row 0-15    │    ...     │  row 0-15    │           │
│  │  col 0-15    │    │  col 16-31   │            │  col 112-127 │           │
│  │              │    │              │            │              │           │
│  └──────────────┘    └──────────────┘            └──────────────┘           │
│                                                                              │
│  Address:  base        base+256B       ...        base+7×256B               │
│            (16×16×1B)                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Total size: 8 blocks × 256B = 2048B = 2KB (for bf16: 16×128×2B = 4KB)
```

### 1.3 Cut-M in NZ Layout: Non-Contiguous Address Problem

When cutting along M (rows), **each AIV writes to non-contiguous addresses**:

```
Cut-M: AIV0 writes rows 0-7, AIV1 writes rows 8-15

Inside EACH [M0=16][N0=16] block:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    Block n1 (any of the 8 blocks):                                         │
│    ┌─────────────────────────────┐                                         │
│    │  row 0  ─────────────────── │  ← AIV0 writes here                     │
│    │  row 1  ─────────────────── │  ← AIV0                                 │
│    │  row 2  ─────────────────── │  ← AIV0                                 │
│    │  row 3  ─────────────────── │  ← AIV0                                 │
│    │  row 4  ─────────────────── │  ← AIV0                                 │
│    │  row 5  ─────────────────── │  ← AIV0                                 │
│    │  row 6  ─────────────────── │  ← AIV0                                 │
│    │  row 7  ─────────────────── │  ← AIV0                                 │
│    │─────────────────────────────│                                         │
│    │  row 8  ─────────────────── │  ← AIV1 writes here                     │
│    │  row 9  ─────────────────── │  ← AIV1                                 │
│    │  row 10 ─────────────────── │  ← AIV1                                 │
│    │  row 11 ─────────────────── │  ← AIV1                                 │
│    │  row 12 ─────────────────── │  ← AIV1                                 │
│    │  row 13 ─────────────────── │  ← AIV1                                 │
│    │  row 14 ─────────────────── │  ← AIV1                                 │
│    │  row 15 ─────────────────── │  ← AIV1                                 │
│    └─────────────────────────────┘                                         │
│                                                                             │
│    Each row = 16 elements × 2B = 32B                                       │
│    AIV0 offset within block: 0                                             │
│    AIV1 offset within block: 8 rows × 32B = 256B (half the block)          │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**AIV0 writes to**: `L1_base + n1 × block_size + 0` for each of 8 blocks
**AIV1 writes to**: `L1_base + n1 × block_size + 8 × row_size` for each of 8 blocks

This is **NOT a simple contiguous copy** — requires strided writes across all N1 blocks!

### 1.4 Strided Write Pattern for Cut-M

```
AIV0 UB [8, 128] → L1 MatTile [16, 128] (NZ layout)

AIV0 must write:
  - To each of 8 blocks (n1 = 0..7)
  - Within each block: rows 0-7 (offset 0, size 8×16×2B = 256B)
  - Stride between blocks: block_size = 16×16×2B = 512B

  L1 addresses for AIV0:
    Block 0: L1_base + 0×512 + 0     = L1_base
    Block 1: L1_base + 1×512 + 0     = L1_base + 512
    Block 2: L1_base + 2×512 + 0     = L1_base + 1024
    ...
    Block 7: L1_base + 7×512 + 0     = L1_base + 3584

AIV1 must write:
  - To each of 8 blocks (n1 = 0..7)  
  - Within each block: rows 8-15 (offset 256B, size 8×16×2B = 256B)
  - Stride between blocks: block_size = 512B

  L1 addresses for AIV1:
    Block 0: L1_base + 0×512 + 256   = L1_base + 256
    Block 1: L1_base + 1×512 + 256   = L1_base + 768
    Block 2: L1_base + 2×512 + 256   = L1_base + 1280
    ...
    Block 7: L1_base + 7×512 + 256   = L1_base + 3840
```

**Generalized formula**:
```
AIV_offset(aiv_id) = aiv_id × (M0/2) × N0 × element_size
                   = aiv_id × 8 × 16 × 2B
                   = aiv_id × 256B

L1_addr(aiv_id, n1) = L1_base + n1 × block_size + AIV_offset(aiv_id)
```

---

## 2. The Problem: Assemble IR Has No Direct ISA

### 2.1 PFA4 AIC Code Pattern

From PA4 pass dump, the AIC function receives data from AIV via `tpop_from_aiv` + `assemble`:

```python
@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0_aic(...):
    pl.comm.aic_initialize_pipe()
    
    # Pop Q tile from AIV (two separate halves)
    qi_0__h0 = pl.comm.tpop_from_aiv(0)  # From AIV0 UB: [8, 128]
    qi_0__h1 = pl.comm.tpop_from_aiv(1)  # From AIV1 UB: [8, 128]
    
    # Assemble into single MatTile for CUBE
    qi_0 = pl.tensor.assemble(
        pl.tensor.assemble(qi_0__h0, [0, 0], [8, 128]),
        [8, 0], [16, 128],
        qi_0__h1
    )  # Result: [16, 128] MatTile in L1
    
    pl.comm.tfree_to_aiv(0)
    pl.comm.tfree_to_aiv(1)
    
    # ... MatMul uses qi_0 from L1 ...
    sij_0 = pl.tensor.matmul(qi_0, kj_0, ...)
```

### 2.2 The Assembly Problem

```
PTO-IR Level (Consumer expresses intent):
─────────────────────────────────────────────────────────────────

    tpop_from_aiv(0)          tpop_from_aiv(1)
           │                         │
           ▼                         ▼
    qi_0__h0 [8,128]          qi_0__h1 [8,128]
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
                   assemble()
                        │
                        ▼
               qi_0 [16, 128] in L1
               (NZ: [8][1][16][16])


What we need at ISA Level (Producer does the work):
─────────────────────────────────────────────────────────────────

    AIV0 UB [8,128]           AIV1 UB [8,128]
           │                         │
           │ tpush_to_aic(0)         │ tpush_to_aic(1)
           │                         │
           │ ub2l1 strided           │ ub2l1 strided
           │ dst=[8][1][8][16]       │ dst=[8][1][8+8×1][16]
           │                         │
           ▼                         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    L1 MatTile [16, 128]                      │
    │                    NZ: [8][1][16][16]                        │
    │                                                              │
    │  Each of 8 blocks:                                          │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  Block n1:  [M0=16][N0=16]                          │    │
    │  │  ┌───────────────────────────────────────────────┐  │    │
    │  │  │  AIV0: rows 0-7   (offset 0)                  │  │    │
    │  │  ├───────────────────────────────────────────────┤  │    │
    │  │  │  AIV1: rows 8-15  (offset 256B)               │  │    │
    │  │  └───────────────────────────────────────────────┘  │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
```

**Key Insight**: The `tpop` + `assemble` on consumer (AIC) side must be **conveyed to the producer (AIV)** so that `tpush_to_aic` can emit the correct strided writes.

---

## 3. Solution: FIFO Definition with Combined Entry for Merged Semantics

### 3.1 Design Principle

The **FIFO entry in L1 must be combined for AIV0 and AIV1** to represent the final merged `tpop` + `assemble` semantics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         L1 FIFO Structure                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        FIFO Slot 0                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │            Combined MatTile [16, 128] (NZ layout)                 │  │ │
│  │  │                                                                   │  │ │
│  │  │  ┌─────────────────────┬─────────────────────┐                   │  │ │
│  │  │  │  AIV0 region        │  AIV1 region        │                   │  │ │
│  │  │  │  rows 0-7           │  rows 8-15          │                   │  │ │
│  │  │  │  (strided write)    │  (strided write)    │                   │  │ │
│  │  │  └─────────────────────┴─────────────────────┘                   │  │ │
│  │  │                                                                   │  │ │
│  │  │  Single FIFO index → Single L1 base address                      │  │ │
│  │  │  Both AIVs write to SAME slot with different offsets             │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        FIFO Slot 1                                      │ │
│  │  (same structure)                                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ...                                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 FIFO Definition Carries Assembly Info

```python
# At pipe initialization time
pl.comm.aiv_initialize_pipe(
    fifo_target='AIV_ALL',  # Combined FIFO for both AIVs
    fifo_config={
        'cut_axis': 'M',              # Cut along M dimension
        'mattile_shape': [16, 128],   # Final assembled shape
        'vectile_shape': [8, 128],    # Each AIV's contribution
        'layout': 'NZ',               # [N1][M1][M0][N0] = [8][1][16][16]
    }
)
```

### 3.3 Lowering: tpush Computes Offset from aiv_id

```python
# tpush_to_aic lowering (on AIV side)

def lower_tpush_to_aic(tile, aiv_id, fifo_config):
    """Lower tpush_to_aic to ub2l1 with computed stride."""
    
    # Get FIFO slot (same for both AIVs!)
    fifo_slot = get_current_fifo_slot()
    l1_base = fifo_slot_to_l1_addr(fifo_slot)
    
    # Compute offset based on aiv_id and cut_axis
    if fifo_config['cut_axis'] == 'M':
        # NZ layout: [N1][M1][M0][N0]
        # AIV0: offset 0 within each block
        # AIV1: offset = (M0/2) × N0 × element_size
        M0 = 16
        N0 = 16
        elem_size = 2  # bf16
        aiv_offset = aiv_id * (M0 // 2) * N0 * elem_size  # 0 or 256B
    
    # Emit strided ub2l1
    emit_copy_ubuf_to_cbuf(
        src=UB_addr(aiv_id, fifo_slot),
        dst=l1_base,
        src_shape=fifo_config['vectile_shape'],
        dst_layout='NZ',
        dst_offset_per_block=aiv_offset,  # Different for AIV0 vs AIV1
        dst_block_stride=M0 * N0 * elem_size  # 512B per block
    )
```

### 3.4 Consumer Side: tpop is Sync, assemble Doesn't Lower

```python
# On AIC side, after lowering:

# tpop_from_aiv(0) → lowers to SYNC (wait cross-core / intra-block)
#                    waits for AIV0's strided write to complete
# tpop_from_aiv(1) → lowers to SYNC (wait cross-core / intra-block)
#                    waits for AIV1's strided write to complete
# assemble()       → doesn't lower to anything (data already in correct layout)

# AIC reads from L1_base with full [16, 128] shape
matmul(L1_tile_at(fifo_slot), ...)
```

---

## 4. Summary: How FIFO Enables Fused TINSERT

| Component | Role |
|-----------|------|
| **FIFO Definition** | Carries cut_axis, layout, shapes — shared by producer and consumer |
| **Single FIFO Index** | Both AIV0 and AIV1 use same slot index → same L1 base address |
| **aiv_id** | Determines offset within each NZ block (0 for AIV0, 256B for AIV1) |
| **tpush_to_aic** | Computes strided write pattern from FIFO config + aiv_id |
| **tpop_from_aiv** | Lowers to SYNC (wait cross-core / intra-block) |
| **assemble()** | Doesn't lower to anything — data already in correct layout |

**Performance Goal Achieved**: Direct strided UB→L1 writes that form the assembled tile during transfer (fused TINSERT), without any L1→L1 copy.

---

## 5. Current Restrictions

1. **Cut-M or Cut-N only**: No arbitrary 2D tiling/assembly
2. **NZ layout only**: For CUBE-bound MatTiles
3. **Symmetric split**: AIV0 and AIV1 contribute equal-sized halves
4. **Combined FIFO entry**: AIV0 and AIV1 must push to same slot

---

## 6. Discussion: Extending the Representation

### 6.1 Potential Extensions

| Extension | Complexity | Notes |
|-----------|------------|-------|
| 2D tiling (quadrants) | High | Would need 4 AIVs or multi-pass |
| Asymmetric splits | Medium | Variable offset computation |
| Non-NZ layouts (ZN, ZZ) | Medium | Different stride patterns |
| Cut-K (for weights) | Medium | Different block structure |

### 6.2 Fallback: L1→L1 Copy

If the pass cannot infer how to lower `tpop` + `assemble` to strided `tpush`:

```
Fallback path (NOT performance-optimal):
1. AIV0 pushes to L1_temp0
2. AIV1 pushes to L1_temp1
3. ??? No L1→L1 path exists ???
4. Would need: read back to UB, then re-copy with strides (2× memory traffic)
```

This is why we **must** capture the assembly semantics in the FIFO definition — the fallback is extremely expensive or impossible.

---

## Appendix: PA4 Example with NZ Layout Detail

```python
# PA4: [16, 128] tile in NZ layout = [8][1][16][16]

# AIV0 pushes [8, 128]:
# - Writes rows 0-7 into each of 8 blocks
# - Within each block: offset 0, size 256B (8 rows × 16 cols × 2B)
# - Block stride: 512B

# AIV1 pushes [8, 128]:
# - Writes rows 8-15 into each of 8 blocks
# - Within each block: offset 256B, size 256B
# - Block stride: 512B

# Result in L1:
# [8][1][16][16] = 8 blocks, each 512B, total 4KB
# Each block has rows 0-7 from AIV0, rows 8-15 from AIV1
```
