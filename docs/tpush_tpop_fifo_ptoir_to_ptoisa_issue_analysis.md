# TPUSH/TPOP FIFO Gap Analysis: AIV0/AIV1 Scheduling Challenges

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
│                            ▼ FIXP (Fixed-Point Unit)                      │
│                            │                                              │
│              ┌─────────────┴─────────────┐                                │
│              │     Dual/Single Dst       │                                │
│              │         Switch            │                                │
│              └─────────────┬─────────────┘                                │
│                            │                                              │
│         ┌──────────────────┼──────────────────┐                           │
│         │                  │                  │                           │
│         ▼                  │                  ▼                           │
│  ┌─────────────────┐       │       ┌─────────────────┐                   │
│  │    AIV0 UB      │       │       │    AIV1 UB      │                   │
│  │                 │       │       │                 │                   │
│  │  256KB (A5)     │       │       │  256KB (A5)     │                   │
│  │  0x00000-0x3FFFF│       │       │  0x00000-0x3FFFF│                   │
│  │                 │       │       │                 │                   │
│  │ (Separate bank) │       │       │ (Separate bank) │                   │
│  └─────────────────┘       │       └─────────────────┘                   │
│                            │                                              │
│                   Dual-Dst: Split M or N                                  │
│                   to SAME UB addr in both                                 │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

**Data Path Summary:**
- **L1 → L0A/L0B**: via MTE1 (Memory Transfer Engine)
- **L0A × L0B → L0C**: via CUBE (Matrix Multiply)
- **L0C → UB**: via FIXP (Fixed-Point Unit) with Dual/Single Dst switch

### 1.2 ISA Operations

| PyPTO API | PTO-ISA Lowering | Description |
|-----------|------------------|-------------|
| `tpush_to_aiv(tile, aiv_id)` | `copy_cc_matrix_to_ubuf` (l0c2ub) | L0C → UB transfer |
| `tpop_from_aic(aiv_id)` | Implicit via FIFO consumption | UB receives L0C data |
| `tfree_to_aiv(aiv_id)` | FIFO slot release | Signal buffer available |

### 1.3 Dual/Single Destination Switch (via FIXP)

The FIXP unit provides a **dual/single destination switch** for L0C→UB transfers:

#### Dual-Dst Mode (Split M or N to SAME UB address)

```
                    L0C Tile [16, 128]
                            │
                            ▼
                    ┌───────────────┐
                    │  FIXP Unit    │
                    │  Dual-Dst ON  │
                    │  split_axis=M │
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
                    ↑                       ↑
                    └───────────────────────┘
                         SAME UB address
```

**Key Constraint**: Dual-dst requires **same UB address** in both AIV0 and AIV1 banks.

#### Single-Dst Mode (Separate transfers)

```
                    L0C Tile [16, 128]
                            │
                            ▼
                    ┌───────────────┐
                    │  FIXP Unit    │
                    │  Single-Dst   │
                    └───────┬───────┘
                            │
                            ▼
                    (Two separate ISA instructions)
                            
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   AIV0 UB @ 0x2000    │       │   AIV1 UB @ 0x4000    │
│   Tile [16, 128]      │       │   Tile [16, 128]      │
└───────────────────────┘       └───────────────────────┘
        (different addresses OK, or same address OK)
```

**Benefit of Dual-Dst**: Single ISA instruction, single L0C read, dual UB writes = 2× bandwidth efficiency.

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

---

## 3. The Problem: Pipeline Efficiency with Separate Scheduling

### 3.1 Problematic IR Pattern

Consider a scenario where the pass generates code with AIV0 and AIV1 pushes in **different loop iterations**:

```python
# Problematic pattern - AIV0 and AIV1 in different loop blocks
@pl.function(type=pl.FunctionType.InCore)
def kernel_aic(...):
    for i in pl.range(0, N, 1):
        tile_0 = compute_tile_for_aiv0(i)
        pl.comm.tpush_to_aiv(tile_0, 0)  # Only AIV0 in this iteration
        
    for j in pl.range(0, M, 1):
        tile_1 = compute_tile_for_aiv1(j)
        pl.comm.tpush_to_aiv(tile_1, 1)  # Only AIV1 in this iteration
```

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel_aiv(..., AIV_IDX):
    if AIV_IDX == 0:
        for i in pl.range(0, N, 1):
            data = pl.comm.tpop_from_aic(0)  # AIV0 pops N times
            process(data)
    else:
        for j in pl.range(0, M, 1):
            data = pl.comm.tpop_from_aic(1)  # AIV1 pops M times
            process(data)
```

### 3.2 Pipeline Efficiency Issue (No Deadlock)

**Clarification**: Without synchronization dependencies between AIV0 and AIV1, there is **no deadlock**. However, there are **pipeline efficiency problems**:

```
Timeline with Separate Scheduling:
────────────────────────────────────────────────────────────────

Scenario: V→C→V pipelined workload (AIV0 pre-processing, AIV1 post-processing)

Optimal (paired dual-dst):
  AIV0: preprocess ──┐
                     ├──→ AIC: matmul ──→ dual-dst push ──┬──→ AIV0: postA
                     │                                    └──→ AIV1: postB
  AIV1: preprocess ──┘

Suboptimal (separate scheduling):
  AIV0: preprocess ──→ AIC: matmul ──→ push(aiv0) ──→ AIV0: postA
  AIV1: preprocess ──→ AIC: matmul ──→ push(aiv1) ──→ AIV1: postB
  
  ❌ Cannot pipeline: Second matmul waits for first to complete
  ❌ 2× L0C reads instead of 1× (no dual-dst)
  ❌ FIFO slot usage inefficient
```

### 3.3 Why Current Design Has This Gap

The `ExpandMixedKernel` pass generates `tpush_to_aiv(tile, aiv_id)` calls that:

1. **Are generic**: Can target AIV0 or AIV1 independently
2. **Are loop-agnostic**: Can appear in any loop structure
3. **Assume shared FIFO**: All pushes go to same FIFO with slot tags

But the lowering to `copy_cc_matrix_to_ubuf` (l0c2ub) ISA:

1. **Cannot detect paired patterns**: When pushes ARE paired, no way to identify and use dual-dst
2. **UB address mismatch**: Separate scheduling may allocate different UB addresses, breaking dual-dst requirement

---

## 4. Proposed Solutions

### 4.1 Option A: Separate AIV0/AIV1 FIFOs

**Design**: Explicit separate FIFOs for AIV0 and AIV1

```python
# New API with explicit FIFO targeting
pl.comm.tpush_to_aiv_fifo(tile, aiv_id=0, fifo_id=0)  # AIV0 FIFO
pl.comm.tpush_to_aiv_fifo(tile, aiv_id=1, fifo_id=1)  # AIV1 FIFO

# Pop from specific FIFO
data_0 = pl.comm.tpop_from_aic_fifo(fifo_id=0)  # AIV0
data_1 = pl.comm.tpop_from_aic_fifo(fifo_id=1)  # AIV1
```

**Pros**:
- ✅ Flexible scheduling: Loops can target AIV0/AIV1 separately
- ✅ Simple mental model
- ✅ Independent progress for each AIV

**Cons**:
- ❌ **Cannot use dual-dst optimization**: Separate FIFOs prevent detecting paired pushes
- ❌ More hardware resources (2 FIFOs vs 1)
- ❌ Pass cannot easily analyze cross-FIFO dependencies

### 4.2 Option B: Dual-Entry FIFO with Same-Sequence Constraint

**Design**: Single FIFO, but paired entries (AIV0, AIV1) must appear together in same sequence

```python
# Constraint: tpush_to_aiv(tile, 0) and tpush_to_aiv(tile, 1) must be adjacent
# and from same source tile (for dual-dst lowering)

# Valid pattern (can lower to dual-dst):
half0 = pl.tensor.view(tile, [8, 128], [0, 0])
half1 = pl.tensor.view(tile, [8, 128], [8, 0])
pl.comm.tpush_to_aiv(half0, 0)  # Must be immediately followed by...
pl.comm.tpush_to_aiv(half1, 1)  # ...push to AIV1 from same tile

# Invalid pattern (error or fallback to non-optimal):
for i in range(N):
    pl.comm.tpush_to_aiv(tile_i, 0)  # AIV0 only - ERROR or separate FIFO fallback
```

**Pros**:
- ✅ **Enables dual-dst ISA**: Paired pushes detected at IR level
- ✅ Single FIFO, less hardware
- ✅ Clear optimization opportunity

**Cons**:
- ❌ Restrictive: Not all algorithms have paired AIV0/AIV1 patterns
- ❌ Requires pass to verify adjacency constraint
- ❌ Fallback needed for non-paired cases

### 4.3 Option C: Explicit SIMD Dual-Entry API (Recommended)

**Design**: New API distinguishing single-AIV vs dual-AIV pushes

```python
# Single-AIV push (uses separate per-AIV FIFO internally)
pl.comm.tpush_to_aiv_single(tile, aiv_id=0)  # Only AIV0, independent FIFO
pl.comm.tpush_to_aiv_single(tile, aiv_id=1)  # Only AIV1, independent FIFO

# Dual-AIV push (uses shared FIFO + dual-dst ISA)
# Requires same UB address in both AIV banks
pl.comm.tpush_to_aiv_dual(
    tile_half0,      # Goes to AIV0 UB
    tile_half1,      # Goes to AIV1 UB
    ub_addr=0x1000,  # Same UB address for both (required for dual-dst)
    split_axis='M'   # M or N axis split
)

# Pop API unchanged (FIFO determined by push type)
data = pl.comm.tpop_from_aic(AIV_IDX)
```

**Lowering**:
```
tpush_to_aiv_single(tile, 0) → copy_cc_matrix_to_ubuf(tile, dst=AIV0_UB[addr0])
tpush_to_aiv_single(tile, 1) → copy_cc_matrix_to_ubuf(tile, dst=AIV1_UB[addr1])

tpush_to_aiv_dual(h0, h1, addr, 'M') → copy_cc_matrix_to_ubuf_dual(
    src=L0C_tile,
    dst0=AIV0_UB[addr],
    dst1=AIV1_UB[addr],  // Same address required
    split='M'
)  // Single ISA instruction, dual write
```

**Pros**:
- ✅ **Best of both**: Flexibility (single) + optimization (dual)
- ✅ Explicit intent: User/pass clearly specifies which mode
- ✅ Dual-dst enabled: Dual-AIV uses optimized ISA with same UB address

**Cons**:
- ❌ API complexity: Two push variants
- ❌ Pass must decide which to emit

---

## 5. IR Pattern Analysis: When to Use Dual vs Single

### 5.1 Detection Heuristic for ExpandMixedKernel Pass

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
                    # ✅ Can use dual-dst
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

### 5.2 PA4 Example - Dual-Dst Candidate

From the PA4 dump:

```python
# This pattern IS dual-dst eligible:
__half0__ = pl.tensor.view(sij_0, [8, 128], [0, 0])    # M split, upper
__half1__ = pl.tensor.view(sij_0, [8, 128], [8, 0])    # M split, lower
pl.comm.tpush_to_aiv(__half0__, 0)  # Adjacent
pl.comm.tpush_to_aiv(__half1__, 1)  # Adjacent

# Analysis:
# - Same source tile: sij_0 [16, 128] from L0C
# - Complementary views: [0:8, :] and [8:16, :] 
# - Adjacent in IR: Yes
# - Can allocate same UB address for both
# → Lower to: tpush_to_aiv_dual(__half0__, __half1__, ub_addr, 'M')
```

### 5.3 Non-Paired Case

```python
# This pattern CANNOT use dual-dst:
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
# - May need different UB addresses
# → Must use: tpush_to_aiv_single(tile, aiv_id) with separate addressing
```

---

## 6. Summary

| Aspect | Current State | Gap | Recommendation |
|--------|---------------|-----|----------------|
| **API** | `tpush_to_aiv(tile, aiv_id)` | Too generic, doesn't distinguish single vs dual | Add `tpush_to_aiv_single` + `tpush_to_aiv_dual` |
| **FIFO** | Assumed shared | Separate scheduling loses dual-dst opportunity | Mode selection: single (per-AIV) vs dual (shared) |
| **Dual-Dst ISA** | Not utilized | Adjacent paired pushes not detected | Pattern detection in pass + dual-dst lowering |
| **UB Address** | Independently allocated | Dual-dst requires same address | Constrain address allocation for paired pushes |
| **Pass Analysis** | Basic block level | Cannot detect cross-loop pairing | Add data flow analysis for push patterns |

### Key Takeaway

The fundamental tension is:

> **Flexibility** (arbitrary AIV0/AIV1 scheduling) vs **Optimization** (dual-dst ISA requiring paired pushes + same UB address)

**Note**: This is primarily a **performance/efficiency** issue, not a correctness/deadlock issue (assuming no AIV0↔AIV1 sync dependencies).

**Recommended approach**: Make this explicit in the API. Let the pass choose single vs dual mode based on pattern analysis, with clear semantics for each.

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

The `view` + consecutive `tpush_to_aiv` pattern with complementary M-axis splits is the ideal candidate for dual-dst ISA lowering.
