# tpush_to_aiv and tpop_from_aic FIFO PTO-IR Design to PTO-ISA Issue Analysis: Separated AIV Calls to Dual-Dst Mapping Challenges

## Executive Summary

This document analyzes a design gap in PyPTO's `ExpandMixedKernel` pass when lowering TPUSH/TPOP operations to the PTO-ISA library. The core issue is the tension between:

1. **Flexible scheduling**: Allowing separate AIV0/AIV1 push operations in different loop iterations
2. **Hardware optimization**: Utilizing dual-destination ISA for efficient L0C→UB transfers
3. **Pipeline efficiency**: Ensuring V→C→V pipelined workloads can be fully scheduled

**Note**: Without synchronization dependencies between AIV0 and AIV1, there is no deadlock risk—only **performance/pipeline efficiency** issues when separate FIFO scheduling prevents optimal dual-dst ISA utilization.

---

## 1. Background: TPUSH/TPOP Architecture

### 1.1 Hardware Model (A5 Architecture)

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              AI Core                                       │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                            L1 Buffer                                 │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                         │
│                                  ▼ MTE1 (Memory Transfer Engine)           │
│                    ┌─────────────┴─────────────┐                          │
│                    │                           │                          │
│                    ▼                           ▼                          │
│  ┌─────────────────────────┐     ┌─────────────────────────┐             │
│  │         L0A             │     │         L0B             │             │
│  │       (Left Mat)        │     │       (Right Mat)       │             │
│  └───────────┬─────────────┘     └─────────────┬───────────┘             │
│              │                                 │                          │
│              └─────────────┬───────────────────┘                          │
│                            │                                              │
│                            ▼ CUBE (Matrix Multiply Unit)                  │
│                            │                                              │
│              ┌─────────────┴─────────────┐                                │
│              │          L0C              │                                │
│              │      (Accumulator)        │                                │
│              └─────────────┬─────────────┘                                │
│                            │                                              │
│                            ▼ FIX-PIPE Unit                                │
│                            │                                              │
│              ┌─────────────┴─────────────┐                                │
│              │     Dual/Single Dst       │                                │
│              │         Switch            │                                │
│              │                           │                                │
│              │  Dual-Dst: Split M or N   │                                │
│              │  to SAME UB addr in both  │                                │
│              └─────────────┬─────────────┘                                │
│                            │                                              │
│         ┌──────────────────┴──────────────────┐                           │
│         │                                     │                           │
│         ▼                                     ▼                           │
│  ┌─────────────────┐               ┌─────────────────┐                   │
│  │    AIV0 UB      │               │    AIV1 UB      │                   │
│  │                 │               │                 │                   │
│  │  256KB (A5)     │               │  256KB (A5)     │                   │
│  │  0x00000-0x3FFFF│               │  0x00000-0x3FFFF│                   │
│  │                 │               │                 │                   │
│  │ (Local addr     │               │ (Local addr     │                   │
│  │  space)         │               │  space)         │                   │
│  └─────────────────┘               └─────────────────┘                   │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

**Data Path Summary:**
- **L1 → L0A/L0B**: via MTE1 (Memory Transfer Engine)
- **L0A × L0B → L0C**: via CUBE (Matrix Multiply)
- **L0C → UB**: via FIX-PIPE Unit with Dual/Single Dst switch

### 1.2 ISA Operations

| PyPTO API | PTO-ISA Lowering | Description |
|-----------|------------------|-------------|
| `tpush_to_aiv(tile, aiv_id)` | `copy_cc_matrix_to_ubuf` (l0c2ub) | L0C → UB transfer |
| `tpop_from_aic(aiv_id)` | Implicit via FIFO consumption | UB receives L0C data |
| `tfree_to_aiv(aiv_id)` | FIFO slot release | Signal buffer available |

### 1.3 Dual/Single Destination Switch (via FIX-PIPE)

The FIX-PIPE unit provides a **dual/single destination switch** for L0C→UB transfers:

#### Dual-Dst Mode (Split M or N to SAME UB address)

```
                    L0C Tile [16, 128]
                            │
                            ▼
                    ┌───────────────┐
                    │ FIX-PIPE Unit │
                    │ Dual-Dst ON   │
                    │ split_axis=M  │
                    │               │
                    │ Split M or N  │
                    │ to SAME addr  │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   AIV0 UB @ 0x1000    │       │   AIV1 UB @ 0x1000    │
│   Tile [8, 128]       │       │   Tile [8, 128]       │
│   (upper half M)      │       │   (lower half M)      │
└───────────────────────┘       └───────────────────────┘
            ↑                               ↑
            └───────── SAME UB address ─────┘
```

**Key Constraint**: Dual-dst requires **same UB address** in both AIV0 and AIV1 local address spaces.

#### Single-Dst Mode (Separate transfers)

```
                    L0C Tile [16, 128]
                            │
                            ▼
                    ┌───────────────┐
                    │ FIX-PIPE Unit │
                    │ Single-Dst    │
                    └───────┬───────┘
                            │
                            ▼
                    (Two separate ISA instructions)
                            
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   AIV0 UB @ 0x2000    │       │   AIV1 UB @ 0x4000    │
│   Tile [8, 128]       │       │   Tile [8, 128]       │
│   (upper half M)      │       │   (lower half M)      │
└───────────────────────┘       └───────────────────────┘
        (different addresses OK, or same address OK)
```

**Performance Comparison**:
- **Dual-Dst**: Single ISA instruction, single L0C read, dual UB writes = **2× bandwidth efficiency**
- **Single-Dst**: Two ISA instructions, two L0C reads, two UB writes = **1/2 performance of dual-dst**

---

## 2. Current PyPTO IR Pattern (Pass 08)

### 2.1 ExpandMixedKernel Output for PA4

From the PA4 (Paged Attention) pass dump after `ExpandMixedKernel`:

```python
# AIC (Cube Unit) - tpush_to_aiv pattern
@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0_aic(...):
    # ... MatMul produces sij[16,128] in L0C ...
    
    # Split tile and push to each AIV separately
    __half0__ = pl.tensor.view(sij_0, [8, 128], [0, 0])    # Upper half
    __half1__ = pl.tensor.view(sij_0, [8, 128], [8, 0])    # Lower half
    
    pl.comm.tpush_to_aiv(__half0__, 0)  # Push to AIV0
    pl.comm.tpush_to_aiv(__half1__, 1)  # Push to AIV1
```

```python
# AIV (Vector Unit) - tpop_from_aic pattern
@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0_aiv(..., AIV_IDX: pl.Scalar[pl.INDEX]):
    # Pop result from AIC (each AIV receives its half)
    sij_0 = pl.comm.tpop_from_aic(AIV_IDX)  # AIV_IDX ∈ {0, 1}
    pl.comm.tfree_to_aic(AIV_IDX)
    
    # Vector processing...
    scaled_0 = pl.tensor.mul(sij_valid_0, scale_0)
```

### 2.2 Current FIFO Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared AIC↔AIV FIFO                       │
│                                                              │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                 │
│  │ S0 │ S1 │ S2 │ S3 │ S4 │ S5 │ S6 │ S7 │  8 slots        │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                 │
│    ▲                                   ▲                    │
│    │ tpush_to_aiv(tile, 0)            │ tpush_to_aiv(tile, 1)
│    │ tpush_to_aiv(tile, 1)            │                    │
│                                                              │
│  Producer (AIC) writes sequentially                         │
│  Consumer (AIV0, AIV1) reads must match order               │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Combined FIFO Index Problem

The problem with a combined (shared) FIFO index is:

1. **AIC loop schedule complexity**: AIC needs to know the schedule differences between AIV0 and AIV1 to compute the combined FIFO index correctly
2. **AIV local index computation**: Each AIV (running as separate SU/control thread) also needs to know **all branches** to compute the combined index correctly from its local loop

**Example**: If AIV0 pushes in iterations [0,2,4] and AIV1 pushes in iterations [1,3,5]:
- AIC must interleave: push(aiv0, idx=0), push(aiv1, idx=1), push(aiv0, idx=2), ...
- AIV0 must compute: "my local iteration 0 → combined index 0", "my local iteration 1 → combined index 2", ...
- AIV1 must compute: "my local iteration 0 → combined index 1", "my local iteration 1 → combined index 3", ...

This requires cross-SU schedule awareness, which breaks the independent control thread model.

### 2.4 Separated FIFO Model (Alternative)

To support separated AIV0/AIV1 representation (non-paired pushes), the design would need **two separate FIFOs with independent indices**:

```
┌─────────────────────────────────────────────────────────────┐
│              Separated FIFO Model (if needed)                │
│                                                              │
│  AIV0 FIFO:                                                  │
│  ┌────┬────┬────┬────┐                                      │
│  │ S0 │ S1 │ S2 │ S3 │  4 slots, index 0-3                  │
│  └────┴────┴────┴────┘                                      │
│    ▲ tpush_to_aiv(tile, 0) only                             │
│                                                              │
│  AIV1 FIFO:                                                  │
│  ┌────┬────┬────┬────┐                                      │
│  │ S0 │ S1 │ S2 │ S3 │  4 slots, index 0-3                  │
│  └────┴────┴────┴────┘                                      │
│    ▲ tpush_to_aiv(tile, 1) only                             │
│                                                              │
│  Independent progress, no cross-AIV ordering dependency     │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 Current Workaround (User Restrictions)

The current design can be kept as-is with the following **user restrictions**:

1. **Single-AIV mode**: Use only AIV0 OR only AIV1 (not both) throughout the kernel
2. **Paired-AIV mode**: If using both AIV0 and AIV1, **must do push/pop for both AIV0 and AIV1 in the same basic block**

```python
# ✅ Valid: Single-AIV mode (only AIV0)
for i in pl.range(0, N, 1):
    pl.comm.tpush_to_aiv(tile, 0)  # Only AIV0, OK

# ✅ Valid: Paired-AIV mode (both in same basic block)
for i in pl.range(0, N, 1):
    pl.comm.tpush_to_aiv(half0, 0)  # AIV0 in same block
    pl.comm.tpush_to_aiv(half1, 1)  # AIV1 in same block, OK

# ❌ Invalid: Mixed mode (AIV0 and AIV1 in different blocks/iterations)
for i in pl.range(0, N, 1):
    if i % 2 == 0:
        pl.comm.tpush_to_aiv(tile, 0)  # AIV0 only
    else:
        pl.comm.tpush_to_aiv(tile, 1)  # AIV1 only - BREAKS combined index
```

This restriction ensures the combined FIFO index can be computed locally without cross-SU schedule awareness.

### 2.6 Current Design Gap

The `ExpandMixedKernel` pass generates `tpush_to_aiv(tile, aiv_id)` calls that:

1. **Are generic**: Can target AIV0 or AIV1 independently
2. **Are loop-agnostic**: Can appear in any loop structure
3. **Assume shared FIFO**: All pushes go to same FIFO with slot tags

The pass CAN detect paired patterns by analyzing cut-M/cut-N view patterns within a single basic block and translate to dual-dst calls. However, the key constraint is:

**UB address alignment**: To use dual-dst, AIV0 and AIV1 must write to the **same UB address**. This requires enforcing a **single FIFO index** for both AIV0 and AIV1 UB FIFOs—i.e., paired pushes must use the same slot index, which guarantees the same UB address allocation in both banks.

---

## 3. Proposed Solutions

### 3.1 Option A: Unified API with aiv_id Enum (Recommended)

**Design**: Single API with `aiv_id` enum: `0` (AIV0), `1` (AIV1), or `-1` (dual/SIMD)

```python
# Pipe initialization specifies FIFO target
# This determines UB allocation and address mapping
pl.comm.initialize_pipe(
    fifo_target='AIV0'     # AIV0 only: allocates AIV0 UB FIFO
    # OR
    fifo_target='AIV1'     # AIV1 only: allocates AIV1 UB FIFO
    # OR
    fifo_target='AIV_ALL'     # SIMD (AIV0||AIV1): allocates paired FIFO with same UB addr
)

# Single API with aiv_id enum
pl.comm.tpush_to_aiv(tile, aiv_id=0)   # AIV0 only (requires fifo_target='AIV0' or 'AIV_ALL')
pl.comm.tpush_to_aiv(tile, aiv_id=1)   # AIV1 only (requires fifo_target='AIV1' or 'AIV_ALL')
pl.comm.tpush_to_aiv(tile, aiv_id=-1)  # Dual (AIV_ALL) (requires fifo_target='AIV_ALL')

# Pop API unchanged
data = pl.comm.tpop_from_aic(AIV_IDX)
```

**FIFO Target in Pipe Initialization**:
- `fifo_target='AIV0'`: Allocates AIV0 UB FIFO only, AIV1 UB not used
- `fifo_target='AIV1'`: Allocates AIV1 UB FIFO only, AIV0 UB not used
- `fifo_target='AIV_ALL'`: Allocates paired FIFO with **same UB address** in both AIV0 and AIV1 banks (required for dual-dst)

**Why FIFO Target Matters**:
- UB allocation and address mapping are bound to pipe initialization
- `SIMD` mode guarantees same UB address for dual-dst ISA
- Separated modes (`AIV0`/`AIV1`) allow independent UB allocation

**Cut-M or Cut-N Inference**:
- Can be inferred from the pop tile shape
- Or initialized in `initialize_pipe()` configuration

**Lowering**:
```
fifo_target='AIV0':
  tpush_to_aiv(tile, 0) → copy_cc_matrix_to_ubuf(tile, dst=AIV0_UB[slot])

fifo_target='AIV1':
  tpush_to_aiv(tile, 1) → copy_cc_matrix_to_ubuf(tile, dst=AIV1_UB[slot])

fifo_target='AIV_ALL':
  tpush_to_aiv(tile, -1) → copy_cc_matrix_to_ubuf_dual(
      src=L0C_tile,
      dst0=AIV0_UB[slot],   // Same slot index
      dst1=AIV1_UB[slot],   // Same slot index → same UB addr
      split=inferred_from_tile
  )
```

**Pros**:
- ✅ Simple unified API with enum
- ✅ Pipe initialization declares FIFO target (AIV0/AIV1/SIMD) upfront
- ✅ UB allocation and address mapping bound at init time
- ✅ AIV_ALL mode enables dual-dst with guaranteed same UB address

**Cons**:
- ❌ Pipe initialization must specify FIFO target before kernel execution

### 3.2 Option B: Fallback for Missed AIV in Basic Block (Not Recommended)

**Design**: If a basic block has only AIV0 or only AIV1 push (but kernel uses both), handle as fallback

```python
# Scenario: Basic block misses one AIV
for i in pl.range(0, N, 1):
    if some_condition:
        pl.comm.tpush_to_aiv(tile, 0)  # Only AIV0, AIV1 missing
    else:
        pl.comm.tpush_to_aiv(tile, 1)  # Only AIV1, AIV0 missing
```

**Possible Fallback Strategies**:
1. **Insert dummy push**: Push zero/padding tile to missing AIV to maintain paired sequence
2. **Dynamic FIFO switch**: Runtime switch between separated/shared FIFO based on pattern
3. **Error/warning**: Reject pattern and require user to restructure code

**Why Not Recommended**:
- Adds complexity without clear benefit
- Dummy pushes waste bandwidth
- Dynamic switching complicates hardware/lowering
- Better to enforce workaround restrictions (Section 2.5)

---

## 4. IR Pattern Analysis

### 4.1 Detection Heuristic for ExpandMixedKernel Pass

```python
def analyze_tpush_pattern(basic_block):
    """Analyze if AIV0 and AIV1 pushes can be paired for dual-dst."""
    
    pushes = find_tpush_ops(basic_block)
    
    # Group by source tile
    for src_tile, push_list in group_by_source(pushes):
        aiv0_pushes = [p for p in push_list if p.aiv_id == 0]
        aiv1_pushes = [p for p in push_list if p.aiv_id == 1]
        
        if len(aiv0_pushes) == 1 and len(aiv1_pushes) == 1:
            p0, p1 = aiv0_pushes[0], aiv1_pushes[0]
            
            # Check if views are complementary halves
            if is_complementary_split(p0.view, p1.view, src_tile):
                # Check if adjacent in IR (no ops between)
                if is_adjacent(p0, p1):
                    # ✅ Can use dual-dst (aiv_id=-1)
                    yield DualPushPattern(p0, p1, src_tile)
                else:
                    # ⚠️ Same block but not adjacent - try to reorder
                    yield ReorderableDualPattern(p0, p1)
            else:
                # ❌ Different splits or different source regions
                yield SinglePushPattern(p0)
                yield SinglePushPattern(p1)
        else:
            # ❌ Unbalanced pushes - use single mode
            for p in push_list:
                yield SinglePushPattern(p)
```

### 4.2 PA4 Example - Dual-Dst Candidate

From the PA4 dump:

```python
# This pattern IS dual-dst eligible (aiv_id=-1):
__half0__ = pl.tensor.view(sij_0, [8, 128], [0, 0])    # M split, upper
__half1__ = pl.tensor.view(sij_0, [8, 128], [8, 0])    # M split, lower
pl.comm.tpush_to_aiv(__half0__, 0)  # Adjacent
pl.comm.tpush_to_aiv(__half1__, 1)  # Adjacent

# Analysis:
# - Same source tile: sij_0 [16, 128] from L0C
# - Complementary views: [0:8, :] and [8:16, :] 
# - Adjacent in IR: Yes
# - Can allocate same UB address for both
# → Lower to: tpush_to_aiv(tile, aiv_id=-1)  // Dual mode
```

### 4.3 Non-Paired Case

```python
# This pattern requires separated FIFOs (aiv_id=0 or 1):
for bn in pl.range(0, bn_this_batch, 1):
    if bn % 2 == 0:
        tile_0 = compute_for_aiv0(bn)
        pl.comm.tpush_to_aiv(tile_0, 0)  # Only AIV0
    else:
        tile_1 = compute_for_aiv1(bn)
        pl.comm.tpush_to_aiv(tile_1, 1)  # Only AIV1

# Analysis:
# - Different loop iterations
# - Cannot be made adjacent
# - Different source tiles
# → Must use: initialize_pipe(fifo_mode='separated')
# → Use: tpush_to_aiv(tile, aiv_id=0) and tpush_to_aiv(tile, aiv_id=1)
```

---

## 5. Summary

| Aspect | Current State | Gap | Recommendation |
|--------|---------------|-----|----------------|
| **API** | `tpush_to_aiv(tile, aiv_id)` | No dual mode | Add `aiv_id=-1` for dual/AIV_ALL mode |
| **FIFO Mode** | Implicit shared | No separated option | `initialize_pipe(fifo_mode=...)` |
| **Dual-Dst ISA** | Not utilized | Adjacent paired pushes not detected | Pattern detection + `aiv_id=-1` lowering |
| **Cut-M/Cut-N** | Not specified | Needs explicit config | Infer from pop tile or pipe init |
| **Pass Analysis** | Basic block level | Cannot detect cross-loop pairing | Add data flow analysis |

**Note**: This is primarily a **performance/efficiency** issue, not a correctness/deadlock issue (assuming no AIV0↔AIV1 sync dependencies).

---

## Appendix: PA4 Pass 08 IR Snippet

```python
# From 08_after_ExpandMixedKernel.py - AIC function

@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0_aic(
    query_0: pl.Tensor[[4096, 128], pl.BFLOAT16],
    # ... other params ...
):
    pl.comm.aic_initialize_pipe()
    
    # Pop Q tile from AIV (assembled from two halves)
    qi_0__h0 = pl.comm.tpop_from_aiv(0)
    qi_0__h1 = pl.comm.tpop_from_aiv(1)
    qi_0 = pl.tensor.assemble(
        pl.tensor.assemble(qi_0__h0, [0, 0], [8, 128]),
        [8, 0], [16, 128],
        qi_0__h1
    )
    pl.comm.tfree_to_aiv(0)
    pl.comm.tfree_to_aiv(1)
    
    # ... MatMul: sij = qi @ kj ...
    sij_0 = pl.tensor.matmul(qi_0, kj_0, ...)
    
    # Push result back to AIV (split into two halves)
    # ⭐ THIS IS THE DUAL-DST CANDIDATE PATTERN ⭐
    __half0__ = pl.tensor.view(sij_0, [8, 128], [0, 0])
    __half1__ = pl.tensor.view(sij_0, [8, 128], [8, 0])
    pl.comm.tpush_to_aiv(__half0__, 0)  # Push to AIV0
    pl.comm.tpush_to_aiv(__half1__, 1)  # Push to AIV1
    
    # Second matmul similarly...
    oi_tmp_0 = pl.tensor.matmul(pij_f16_1, vj_0, ...)
    __half0__1 = pl.tensor.view(oi_tmp_0, [8, 128], [0, 0])
    __half1__1 = pl.tensor.view(oi_tmp_0, [8, 128], [8, 0])
    pl.comm.tpush_to_aiv(__half0__1, 0)
    pl.comm.tpush_to_aiv(__half1__1, 1)
```

The `view` + consecutive `tpush_to_aiv` pattern with complementary M-axis splits is the ideal candidate for dual-dst ISA lowering with `aiv_id=-1`.
