# TPUSH/TPOP ISA Design: 1:1 and 1:2 Communication Modes

## Overview

This document specifies the enhanced TPUSH/TPOP instruction design supporting **1:1** (single AIV) and **1:2** (dual AIV) communication modes between Cube and Vector cores.

---

## Communication Modes

| Mode | C2V (Cube→Vector) | V2C (Vector→Cube) |
|------|-------------------|-------------------|
| **1:1** | 1 Cube `tpush_to_aiv` → 1 Vector `tpop_from_aic` | 1 Vector `tpush_to_aic` → 1 Cube `tpop_from_aiv` |
| **1:2** | 1 Cube `tpush_to_aiv` → 2 Vector `tpop_from_aic` | 2 Vector `tpush_to_aic` → 1 Cube `tpop_from_aiv` |

---

## Split/Combine Axis

The tile split/combine axis **must be specified** so kernel code knows which half of the data it receives/sends:

```cpp
enum SplitMode : uint8_t {
    SPLIT_NONE  = 0,   // 1:1 mode, no split
    SPLIT_M     = 1,   // Split along M (rows): AIV0 gets upper half, AIV1 gets lower half
    SPLIT_N     = 2,   // Split along N (cols): AIV0 gets left half, AIV1 gets right half
};

enum CombineMode : uint8_t {
    COMBINE_NONE = 0,  // 1:1 mode, no combine
    COMBINE_M    = 1,  // Combine along M (rows): AIV0 writes upper half, AIV1 writes lower half
    COMBINE_N    = 2,  // Combine along N (cols): AIV0 writes left half, AIV1 writes right half
};
```

---

## C2V Direction: Cube → Vector (Split)

### 1:1 Mode

```
Cube:   tpush_to_aiv(tile, aiv_idx=0)   // Push full tile to AIV0
Vector: tpop_from_aic(tile, aiv_idx=0)  // AIV0 receives full tile
```

### 1:2 Mode with SPLIT_M or SPLIT_N

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        C2V 1:2 with SPLIT_M                                 │
│                                                                             │
│  Cube L1 Tile [16, 128]:                                                   │
│  ┌───────────────────┐                                                     │
│  │ upper [8, 128]    │ ──────► AIV0 UB                                     │
│  ├───────────────────┤                                                     │
│  │ lower [8, 128]    │ ──────► AIV1 UB                                     │
│  └───────────────────┘                                                     │
│                                                                             │
│  Cube:    tpush_to_aiv(tile, SPLIT_M)     // ONE call, waits BOTH free     │
│  Vector0: tpop_from_aic(tile, SPLIT_M)    // receives upper half           │
│  Vector1: tpop_from_aic(tile, SPLIT_M)    // receives lower half           │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                        C2V 1:2 with SPLIT_N                                 │
│                                                                             │
│  Cube L1 Tile [16, 128]:                                                   │
│  ┌──────────┬──────────┐                                                   │
│  │ left     │ right    │                                                   │
│  │[16, 64]  │[16, 64]  │                                                   │
│  └──────────┴──────────┘                                                   │
│       │           │                                                         │
│       ▼           ▼                                                         │
│    AIV0 UB     AIV1 UB                                                     │
│                                                                             │
│  Cube:    tpush_to_aiv(tile, SPLIT_N)     // ONE call, waits BOTH free     │
│  Vector0: tpop_from_aic(tile, SPLIT_N)    // receives left half            │
│  Vector1: tpop_from_aic(tile, SPLIT_N)    // receives right half           │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### `tpush_to_aiv` Instruction (C2V Producer on Cube)

```cpp
// 1:1 mode
void tpush_to_aiv(Tile& tile, uint8_t aiv_idx);

// 1:2 mode - single call, splits to both AIVs
void tpush_to_aiv(Tile& tile, SplitMode split);
```

**1:2 Mode Pseudocode**:
```
function tpush_to_aiv(TILE, SPLIT_MODE):
    // 1) Wait for BOTH AIV0 and AIV1 free flags
    WAIT flag_free[C2V, AIV0]: target_tag
    WAIT flag_free[C2V, AIV1]: target_tag

    // 2) DMA upper/left half to AIV0's slot
    if SPLIT_MODE == SPLIT_M:
        src0 = TILE.data                           // rows 0-7
        src1 = TILE.data + 8 * row_stride          // rows 8-15
    else:  // SPLIT_N
        src0 = TILE.data                           // cols 0-63
        src1 = TILE.data + 64 * element_size       // cols 64-127

    MTE_copy(src=src0, dst=aiv0_slot[target_tag], size=HALF_TILE_SIZE)
    MTE_copy(src=src1, dst=aiv1_slot[target_tag], size=HALF_TILE_SIZE)
    WAIT mte_flags

    // 3) Signal BOTH AIV0 and AIV1 ready
    SET flag_ready[C2V, AIV0]: target_tag
    SET flag_ready[C2V, AIV1]: target_tag

    // 4) Advance tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

### `tpop_from_aic` Instruction (C2V Consumer on Vector)

```cpp
// 1:1 mode
void tpop_from_aic(Tile& tile, uint8_t aiv_idx);

// 1:2 mode - each AIV calls with split mode to know which half it receives
void tpop_from_aic(Tile& tile, SplitMode split);
```

**1:2 Mode Pseudocode** (executed by each AIV):
```
function tpop_from_aic(TILE, SPLIT_MODE):
    aiv_idx = get_my_aiv_idx()   // 0 or 1

    // 1) Wait for my ready flag
    WAIT flag_ready[C2V, aiv_idx]: target_tag

    // 2) Receive my half (zero-copy on A5)
    // The kernel knows: if aiv_idx==0, it has upper/left half
    //                   if aiv_idx==1, it has lower/right half
    TILE.data = my_slot[target_tag]

    // 3) Signal my free flag
    SET flag_free[C2V, aiv_idx]: target_tag

    // 4) Advance tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

---

## V2C Direction: Vector → Cube (Combine)

### 1:1 Mode

```
Vector: tpush_to_aic(tile, aiv_idx=0)  // Push full tile from AIV0
Cube:   tpop_from_aiv(tile, aiv_idx=0) // Cube receives full tile
```

### 1:2 Mode with COMBINE_M or COMBINE_N

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        V2C 1:2 with COMBINE_M                               │
│                                                                             │
│  AIV0 UB [8, 128]  ────┐                                                   │
│                        │      Cube L1 Tile [16, 128]:                      │
│                        ├────► ┌───────────────────┐                        │
│                        │      │ upper [8, 128]    │ ← from AIV0            │
│  AIV1 UB [8, 128]  ────┤      ├───────────────────┤                        │
│                        └────► │ lower [8, 128]    │ ← from AIV1            │
│                               └───────────────────┘                        │
│                                                                             │
│  Vector0: tpush_to_aic(tile, COMBINE_M)  // writes upper half              │
│  Vector1: tpush_to_aic(tile, COMBINE_M)  // writes lower half              │
│  Cube:    tpop_from_aiv(tile, COMBINE_M) // ONE call, waits BOTH ready     │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                        V2C 1:2 with COMBINE_N                               │
│                                                                             │
│  AIV0 UB [16, 64]  ────┐                                                   │
│                        │      Cube L1 Tile [16, 128]:                      │
│                        ├────► ┌──────────┬──────────┐                      │
│                        │      │ left     │ right    │                      │
│  AIV1 UB [16, 64]  ────┤      │[16, 64]  │[16, 64]  │                      │
│                        │      │← AIV0    │← AIV1    │                      │
│                        └────► └──────────┴──────────┘                      │
│                                                                             │
│  Vector0: tpush_to_aic(tile, COMBINE_N)  // writes left half               │
│  Vector1: tpush_to_aic(tile, COMBINE_N)  // writes right half              │
│  Cube:    tpop_from_aiv(tile, COMBINE_N) // ONE call, waits BOTH ready     │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### `tpush_to_aic` Instruction (V2C Producer on Vector)

```cpp
// 1:1 mode
void tpush_to_aic(Tile& tile, uint8_t aiv_idx);

// 1:2 mode - each AIV calls with combine mode to know its write offset
void tpush_to_aic(Tile& tile, CombineMode combine);
```

**1:2 Mode Pseudocode** (executed by each AIV):
```
function tpush_to_aic(TILE, COMBINE_MODE):
    aiv_idx = get_my_aiv_idx()   // 0 or 1

    // 1) Wait for my free flag
    WAIT flag_free[V2C, aiv_idx]: target_tag

    // 2) Compute my write offset in the shared L1 slot
    if COMBINE_MODE == COMBINE_M:
        // AIV0 → offset 0 (rows 0-7), AIV1 → offset rows*8 (rows 8-15)
        dst_offset = aiv_idx * 8 * row_stride
    else:  // COMBINE_N
        // AIV0 → offset 0 (cols 0-63), AIV1 → offset cols*64 (cols 64-127)
        dst_offset = aiv_idx * 64 * element_size

    // 3) DMA my half to the correct offset in shared L1 slot
    dst = cube_l1_slot[target_tag] + dst_offset
    MTE_strided_copy(src=TILE.data, dst=dst, ...)
    WAIT mte_flag

    // 4) Signal my ready flag
    SET flag_ready[V2C, aiv_idx]: target_tag

    // 5) Advance tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

### `tpop_from_aiv` Instruction (V2C Consumer on Cube)

```cpp
// 1:1 mode
void tpop_from_aiv(Tile& tile, uint8_t aiv_idx);

// 1:2 mode - single call, waits for both AIVs, receives combined tile
void tpop_from_aiv(Tile& tile, CombineMode combine);
```

**1:2 Mode Pseudocode**:
```
function tpop_from_aiv(TILE, COMBINE_MODE):
    // 1) Wait for BOTH AIV0 and AIV1 ready flags
    WAIT flag_ready[V2C, AIV0]: target_tag
    WAIT flag_ready[V2C, AIV1]: target_tag

    // 2) The combined tile is now complete in L1 slot
    //    (AIV0 wrote upper/left, AIV1 wrote lower/right)
    TILE.data = l1_slot[target_tag]   // zero-copy on A5

    // 3) Signal BOTH AIV0 and AIV1 free
    SET flag_free[V2C, AIV0]: target_tag
    SET flag_free[V2C, AIV1]: target_tag

    // 4) Advance tag
    target_tag = (target_tag + 1) % SLOT_NUM
```

---

## Initialization API Updates

### `aic_initialize_pipe`

```cpp
void aic_initialize_pipe(
    uint8_t  DIR_MASK,
    uint32_t SLOT_SIZE,
    void*    GM_SLOT_BUFFER,
    uint32_t C2V_CONSUMER_BUF,
    uint32_t V2C_CONSUMER_BUF,
    // NEW parameters:
    SplitMode   C2V_SPLIT,     // SPLIT_NONE, SPLIT_M, or SPLIT_N
    CombineMode V2C_COMBINE    // COMBINE_NONE, COMBINE_M, or COMBINE_N
);
```

### `aiv_initialize_pipe`

```cpp
void aiv_initialize_pipe(
    uint8_t  DIR_MASK,
    uint32_t SLOT_SIZE,
    void*    GM_SLOT_BUFFER,
    uint32_t C2V_CONSUMER_BUF,
    uint32_t V2C_CONSUMER_BUF,
    // NEW parameters:
    SplitMode   C2V_SPLIT,     // SPLIT_NONE, SPLIT_M, or SPLIT_N
    CombineMode V2C_COMBINE    // COMBINE_NONE, COMBINE_M, or COMBINE_N
);
```

---

## API Summary

| Instruction | Core | Mode | Parameters | Behavior |
|-------------|------|------|------------|----------|
| `tpush_to_aiv(tile, aiv_idx)` | Cube | 1:1 | `aiv_idx` | Push full tile to one AIV |
| `tpush_to_aiv(tile, split)` | Cube | 1:2 | `SPLIT_M/SPLIT_N` | Wait both free, split tile to both AIVs |
| `tpop_from_aic(tile, aiv_idx)` | Vector | 1:1 | `aiv_idx` | Receive full tile from Cube |
| `tpop_from_aic(tile, split)` | Vector | 1:2 | `SPLIT_M/SPLIT_N` | Receive my half (upper/left or lower/right) |
| `tpush_to_aic(tile, aiv_idx)` | Vector | 1:1 | `aiv_idx` | Push full tile to Cube |
| `tpush_to_aic(tile, combine)` | Vector | 1:2 | `COMBINE_M/COMBINE_N` | Push my half with strided write |
| `tpop_from_aiv(tile, aiv_idx)` | Cube | 1:1 | `aiv_idx` | Receive full tile from one AIV |
| `tpop_from_aiv(tile, combine)` | Cube | 1:2 | `COMBINE_M/COMBINE_N` | Wait both ready, receive combined tile |

---

## Why Split/Combine Mode Must Be Specified

The kernel code **cannot compute correctly** without knowing the split/combine axis:

```python
# C2V 1:2 Example - Vector kernel
@pl.incore
def vector_kernel(aiv_idx, ...):
    aiv_initialize_pipe(..., C2V_SPLIT=SPLIT_M)

    for ...:
        # tpop tells me which half I have
        half_tile = tpop_from_aic(SPLIT_M)

        # Now I know:
        #   if aiv_idx == 0: half_tile contains rows 0-7
        #   if aiv_idx == 1: half_tile contains rows 8-15
        # My compute code can process correctly!
        result = compute(half_tile)
```

```python
# V2C 1:2 Example - Vector kernel
@pl.incore
def vector_kernel(aiv_idx, ...):
    aiv_initialize_pipe(..., V2C_COMBINE=COMBINE_M)

    for ...:
        # I compute my half
        if aiv_idx == 0:
            my_half = compute_upper_half()   # rows 0-7
        else:
            my_half = compute_lower_half()   # rows 8-15

        # tpush writes to correct offset based on COMBINE_M
        tpush_to_aic(my_half, COMBINE_M)
```

---

## Flow Control Summary

| Direction | Mode | Producer waits | Consumer waits |
|-----------|------|----------------|----------------|
| C2V | 1:1 | `free[aiv_idx]` | `ready[aiv_idx]` |
| C2V | 1:2 | `free[AIV0]` AND `free[AIV1]` | `ready[my_aiv_idx]` |
| V2C | 1:1 | `free[aiv_idx]` | `ready[aiv_idx]` |
| V2C | 1:2 | `free[my_aiv_idx]` | `ready[AIV0]` AND `ready[AIV1]` |
