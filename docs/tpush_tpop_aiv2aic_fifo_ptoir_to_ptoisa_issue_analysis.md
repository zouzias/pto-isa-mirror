# tpush_to_aic and tpop_from_aiv FIFO PTO-IR Design to PTO-ISA Issue Analysis: AIV→AIC Data Assembly Challenges

## Executive Summary

This document analyzes the V→C (Vector to Cube) data path challenges when lowering `tpush_to_aic` / `tpop_from_aiv` operations from PTO-IR to PTO-ISA. The core issue is:

1. **Two separate UB buffers** (AIV0 UB, AIV1 UB) must be assembled into a **single L1 MatTile** for CUBE consumption
2. **No L1→L1 path** exists in current NPU hardware, so `TINSERT`/`assemble` IR cannot directly translate to ISA
3. The solution uses **FIFO definition** to convey cut-M/cut-N semantics from consumer (`tpop_from_aiv` + `assemble`) to producer (`tpush_to_aic`)

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
│           │                                             │                    │
│           ▼                                             ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                            L1 Buffer                                 │    │
│  │                                                                      │    │
│  │   ┌────────────────────────────────────────────────────────────┐    │    │
│  │   │              MatTile [16, 128] (NZ Layout)                  │    │    │
│  │   │                                                             │    │    │
│  │   │    ┌─────────────────┬─────────────────┐                   │    │    │
│  │   │    │  From AIV0 UB   │  From AIV1 UB   │                   │    │    │
│  │   │    │  [0:8, :]       │  [8:16, :]      │  (Cut-M)          │    │    │
│  │   │    └─────────────────┴─────────────────┘                   │    │    │
│  │   │                                                             │    │    │
│  │   │    Physical Layout: [N1][M1][M0=16][N0=16]                  │    │    │
│  │   └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │   ❌ NO L1→L1 PATH: Cannot assemble tiles within L1                 │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│                            │ MTE1 (l12l0)                                    │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         L0A / L0B                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Constraint**: No L1→L1 data path means we cannot do post-hoc assembly in L1. The assembly must be done **during** the UB→L1 copy via strided stores.

### 1.2 NZ Layout for CUBE

For CUBE computation (MatMul), activation tiles use NZ (non-transposed) layout:

```
Physical Memory Layout: [N1][M1][M0=16][N0=16]

Where:
  N = N1 × N0  (e.g., 128 = 8 × 16)
  M = M1 × M0  (e.g., 16 = 1 × 16)

Logical View [M, N] = [16, 128]:
  Row 0-15 (M dimension) × Col 0-127 (N dimension)
```

### 1.3 ISA Operations

| PyPTO API | PTO-ISA Lowering | Description |
|-----------|------------------|-------------|
| `tpush_to_aic(tile, aiv_id)` | `copy_ubuf_to_cbuf` (ub2l1) | UB → L1 transfer |
| `tpop_from_aiv(aiv_id)` | Implicit via FIFO consumption | L1 receives assembled tile |
| `tfree_to_aiv(aiv_id)` | FIFO slot release | Signal buffer available |

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
PTO-IR Level:
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
                        │
                        ▼
              MatMul (CUBE uses NZ layout)


ISA Level Problem:
─────────────────────────────────────────────────────────────────

    AIV0 UB [8,128]           AIV1 UB [8,128]
           │                         │
           │ ub2l1                   │ ub2l1
           │                         │
           ▼                         ▼
    L1 region A               L1 region B
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
              ❌ TINSERT/assemble
              (No L1→L1 path!)
                        │
                        ▼
               Single L1 MatTile?
```

**Problem**: The `assemble` IR operation implies an L1→L1 copy/insert, but **no such hardware path exists**. We cannot directly translate `TINSERT`/`assemble` to ISA.

### 2.3 Diagram: The Gap Between IR and Hardware

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        What IR Expresses                                     │
│                                                                              │
│   AIV0 UB              AIV1 UB                                              │
│   [8,128]              [8,128]                                              │
│      │                    │                                                  │
│      │ tpush_to_aic(0)    │ tpush_to_aic(1)                                 │
│      ▼                    ▼                                                  │
│   L1 temp0             L1 temp1          ← Two separate L1 regions          │
│      │                    │                                                  │
│      └────────┬───────────┘                                                  │
│               │ assemble (TINSERT)                                           │
│               ▼                                                              │
│         L1 MatTile [16,128]              ← Unified tile for CUBE            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     What Hardware Requires                                   │
│                                                                              │
│   AIV0 UB              AIV1 UB                                              │
│   [8,128]              [8,128]                                              │
│      │                    │                                                  │
│      │ ub2l1 (stride)     │ ub2l1 (stride)                                  │
│      │ dst=L1+offset0     │ dst=L1+offset1                                  │
│      ▼                    ▼                                                  │
│   ┌─────────────────────────────────────┐                                   │
│   │        L1 MatTile [16,128]          │  ← Single unified address         │
│   │  ┌────────────┬────────────┐        │                                   │
│   │  │ [0:8,:]    │ [8:16,:]   │        │  ← Strided writes to same tile    │
│   │  │ from AIV0  │ from AIV1  │        │                                   │
│   │  └────────────┴────────────┘        │                                   │
│   └─────────────────────────────────────┘                                   │
│                                                                              │
│   ✅ No L1→L1 needed: Both UB→L1 writes go to same MatTile address         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Current Solution: FIFO Definition with Cut-M/Cut-N Semantics

### 3.1 Design Principle

Since `assemble`/`TINSERT` cannot be executed in hardware, we **convey the assembly semantics through the FIFO definition** at pipe initialization time:

1. **Consumer side** (`tpop_from_aiv` + `assemble`): Defines the desired final MatTile layout and how pieces fit together
2. **Producer side** (`tpush_to_aic`): Uses FIFO info to compute correct L1 offsets for strided stores
3. **FIFO carries**: Cut axis (M or N), tile dimensions, and layout info

### 3.2 FIFO Definition Carries Assembly Info

```python
# At pipe initialization time
pl.comm.aiv_initialize_pipe(
    fifo_config={
        'aiv0_aiv1_to_l1': {
            'cut_axis': 'M',           # Cut along M dimension
            'mattile_shape': [16, 128], # Final assembled shape
            'vectile_shape': [8, 128],  # Each AIV's contribution
            'layout': 'NZ',             # [N1][M1][M0=16][N0=16]
        }
    }
)
```

### 3.3 Lowering Strategy

```
PTO-IR:
    tpush_to_aic(half0, aiv_id=0)  # From AIV0
    tpush_to_aic(half1, aiv_id=1)  # From AIV1

    tpop_from_aiv(0)  # qi_0__h0
    tpop_from_aiv(1)  # qi_0__h1
    assemble(qi_0__h0, qi_0__h1)  # → qi_0

PTO-ISA (with FIFO info):
    # FIFO knows: cut_axis='M', mattile=[16,128], layout=NZ
    
    # AIV0: Write to L1 offset 0 (rows 0-7)
    copy_ubuf_to_cbuf(
        src=AIV0_UB[slot],
        dst=L1_MatTile + offset_for_M_half_0,  # Computed from FIFO
        stride=NZ_stride_for_cut_M
    )
    
    # AIV1: Write to L1 offset for rows 8-15
    copy_ubuf_to_cbuf(
        src=AIV1_UB[slot],
        dst=L1_MatTile + offset_for_M_half_1,  # Computed from FIFO
        stride=NZ_stride_for_cut_M
    )
    
    # assemble() becomes no-op: data already in correct L1 layout
```

### 3.4 Cut-M vs Cut-N in NZ Layout

**NZ Layout**: `[N1][M1][M0=16][N0=16]`

```
Cut-M Example (M=16 split into 2×8):
────────────────────────────────────────────────────────────────

Logical: [16, 128] → AIV0 gets [0:8, :], AIV1 gets [8:16, :]

Physical NZ [N1=8][M1=1][M0=16][N0=16]:
  - M0=16 is the innermost M dimension
  - Cut-M splits M1 or requires interleaved writes

  AIV0 writes: rows 0-7 of each [M0=16][N0=16] block
  AIV1 writes: rows 8-15 of each [M0=16][N0=16] block

  Stride pattern for ub2l1 must account for NZ layout.


Cut-N Example (N=128 split into 2×64):
────────────────────────────────────────────────────────────────

Logical: [16, 128] → AIV0 gets [:, 0:64], AIV1 gets [:, 64:128]

Physical NZ [N1=8][M1=1][M0=16][N0=16]:
  - N1=8 means 8 blocks of N0=16
  - Cut-N splits N1: AIV0 gets N1=[0:4], AIV1 gets N1=[4:8]

  AIV0 writes: first 4 [M0][N0] blocks (cols 0-63)
  AIV1 writes: last 4 [M0][N0] blocks (cols 64-127)

  Simpler stride pattern (contiguous N1 blocks).
```

### 3.5 Inferring Cut Axis from Shape

The cut axis can be inferred from the ratio of VecTile to MatTile dimensions:

```python
def infer_cut_axis(vectile_shape, mattile_shape):
    """Infer cut-M or cut-N from shape ratio."""
    v_rows, v_cols = vectile_shape
    m_rows, m_cols = mattile_shape
    
    if v_rows < m_rows and v_cols == m_cols:
        return 'M'  # Cut along M (rows)
    elif v_rows == m_rows and v_cols < m_cols:
        return 'N'  # Cut along N (cols)
    else:
        raise ValueError("Cannot infer cut axis")

# PA4 example:
infer_cut_axis([8, 128], [16, 128])  # → 'M' (rows halved, cols same)
```

---

## 4. PFA4 Example Walkthrough

### 4.1 Producer Side (AIV)

```python
@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0_aiv(..., AIV_IDX: pl.Scalar[pl.INDEX]):
    # ... vector processing produces q_tile [8, 128] in UB ...
    
    # Push to AIC via FIFO (FIFO knows cut-M semantics)
    pl.comm.tpush_to_aic(q_tile, AIV_IDX)  # AIV_IDX ∈ {0, 1}
```

### 4.2 Consumer Side (AIC)

```python
@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0_aic(...):
    # Pop from FIFO (each pop returns a half)
    qi_0__h0 = pl.comm.tpop_from_aiv(0)  # [8, 128] from AIV0
    qi_0__h1 = pl.comm.tpop_from_aiv(1)  # [8, 128] from AIV1
    
    # Assemble (IR level - conveys intent)
    qi_0 = pl.tensor.assemble(
        pl.tensor.assemble(qi_0__h0, [0, 0], [8, 128]),
        [8, 0], [16, 128],
        qi_0__h1
    )
    
    # CUBE uses assembled tile
    sij_0 = pl.tensor.matmul(qi_0, kj_0, ...)
```

### 4.3 Lowered ISA (Unified L1 with Strided Stores)

```
# FIFO config: cut_axis='M', mattile=[16,128], layout=NZ

# AIV0 tpush_to_aic(q_tile, 0):
copy_ubuf_to_cbuf(
    src = AIV0_UB + fifo_slot_offset,
    dst = L1_MatTile_base,              # Same base for both
    src_shape = [8, 128],
    dst_stride = NZ_M_cut_stride_0,     # Write to rows 0-7
)

# AIV1 tpush_to_aic(q_tile, 1):
copy_ubuf_to_cbuf(
    src = AIV1_UB + fifo_slot_offset,
    dst = L1_MatTile_base,              # Same base as AIV0!
    src_shape = [8, 128],
    dst_stride = NZ_M_cut_stride_1,     # Write to rows 8-15
)

# tpop_from_aiv(0), tpop_from_aiv(1), assemble():
#   → NO-OP at ISA level
#   → Data already assembled in L1 via strided stores

# matmul reads from L1_MatTile_base with full [16,128] shape
```

---

## 5. Current Restrictions

### 5.1 Supported Patterns

The current design restricts representation to:

1. **Cut-M or Cut-N only**: No arbitrary 2D tiling/assembly
2. **NZ layout only**: For CUBE-bound MatTiles
3. **Symmetric split**: AIV0 and AIV1 contribute equal-sized halves
4. **FIFO-based coordination**: Assembly info encoded in FIFO definition

### 5.2 What This Enables

```python
# ✅ Supported: Cut-M (rows split)
AIV0: [8, 128] → L1[0:8, :]
AIV1: [8, 128] → L1[8:16, :]
Result: [16, 128] MatTile

# ✅ Supported: Cut-N (cols split)
AIV0: [16, 64] → L1[:, 0:64]
AIV1: [16, 64] → L1[:, 64:128]
Result: [16, 128] MatTile

# ❌ Not Supported: 2D split (quadrants)
AIV0: [8, 64]  → L1[0:8, 0:64]      # Would need 4 AIVs
AIV1: [8, 64]  → L1[0:8, 64:128]    # or 2-stage assembly
AIV2: [8, 64]  → L1[8:16, 0:64]
AIV3: [8, 64]  → L1[8:16, 64:128]
```

---

## 6. Discussion: Extending the Representation

### 6.1 Potential Extensions

| Extension | Complexity | Hardware Requirement |
|-----------|------------|---------------------|
| 2D tiling (quadrants) | High | More AIVs or multi-pass |
| Asymmetric splits | Medium | Variable stride computation |
| Non-NZ layouts (ZN, ZZ) | Medium | Layout-aware stride tables |
| Runtime cut-axis selection | Low | FIFO config at init time |

### 6.2 Alternative: Explicit TINSERT with Strided Store

If future hardware adds L1→L1 paths, the IR could directly lower `TINSERT`:

```python
# Future possibility (not current hardware)
tinsert(
    dst=L1_MatTile,
    src=L1_temp,
    offset=[8, 0],  # Insert at row 8
    shape=[8, 128]
)
```

### 6.3 Current Best Practice

For now, the recommended pattern is:

1. **Use cut-M or cut-N** based on tile dimensions
2. **Encode in FIFO config** at pipe initialization
3. **Let lowering compute strides** for NZ layout
4. **Avoid explicit assemble** in IR when possible (convey via FIFO)

---

## 7. Summary

| Aspect | Challenge | Solution |
|--------|-----------|----------|
| **Hardware** | No L1→L1 path | Strided UB→L1 stores to unified address |
| **IR Semantics** | `assemble`/`TINSERT` not directly executable | FIFO carries assembly info |
| **Layout** | NZ format for CUBE | Cut-M/Cut-N with computed strides |
| **Coordination** | AIV0/AIV1 must write to same L1 tile | Single FIFO index, shared base address |
| **Restrictions** | Only cut-M or cut-N | Matches V→C fusion patterns in practice |

---

## Appendix: PA4 Pass 08 IR Snippet (V→C Path)

```python
# From 08_after_ExpandMixedKernel.py - AIC function (consumer)

@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0_aic(
    query_0: pl.Tensor[[4096, 128], pl.BFLOAT16],
    # ... other params ...
):
    pl.comm.aic_initialize_pipe()
    
    # ════════════════════════════════════════════════════════════
    # V→C Path: Pop from AIV UB, assemble into L1 MatTile
    # ════════════════════════════════════════════════════════════
    
    # Pop Q tile halves from AIV0 and AIV1
    qi_0__h0 = pl.comm.tpop_from_aiv(0)  # [8, 128] from AIV0 UB
    qi_0__h1 = pl.comm.tpop_from_aiv(1)  # [8, 128] from AIV1 UB
    
    # Assemble into single [16, 128] MatTile
    # ⭐ THIS IS THE ASSEMBLY PATTERN - LOWERED VIA FIFO STRIDES ⭐
    qi_0 = pl.tensor.assemble(
        pl.tensor.assemble(qi_0__h0, [0, 0], [8, 128]),
        [8, 0], [16, 128],
        qi_0__h1
    )
    
    pl.comm.tfree_to_aiv(0)
    pl.comm.tfree_to_aiv(1)
    
    # ════════════════════════════════════════════════════════════
    # CUBE Path: MatMul uses assembled L1 tile
    # ════════════════════════════════════════════════════════════
    
    sij_0 = pl.tensor.matmul(qi_0, kj_0, ...)  # qi_0 is now [16,128] in L1
```

The `tpop_from_aiv` + `assemble` pattern captures the V→C assembly semantics. The FIFO definition conveys this to `tpush_to_aic` on the AIV side, enabling correct strided stores to form the unified L1 MatTile.
