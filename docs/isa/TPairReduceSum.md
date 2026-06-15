# TPAIRREDUCESUM


## Tile Operation Diagram

![TPAIRREDUCESUM](../figures/isa/TPairReduceSum.svg)


## Introduction

Pair-reduction sum: add every two adjacent elements of the source tile and write the results into the lower half of the destination tile. The instruction uses the `vcpadd` vector pair-addition primitive, which consumes `src0[i, 2k] + src0[i, 2k+1]` for each pair index `k`, packing the reduced values into positions `0 … ⌈validCols/2⌉−1` of each destination row. Elements in the upper half of the destination (positions `⌈validCols/2⌉ … validCols−1`) are filled with **0**. Inactive elements (positions outside the valid region) are also treated as **0**.

## Math Interpretation

Given a source tile `src0` and a destination tile `dst` with the same valid shape `(validRows, validCols)`, for each row `i` and pair index `k`:

$$ \mathrm{dst}_{i,k} = \mathrm{src0}_{i, 2k} + \mathrm{src0}_{i, 2k+1}, \quad 0 \le k < \left\lceil \frac{\mathrm{validCols}}{2} \right\rceil $$

Elements at positions `⌈validCols/2⌉ … validCols−1` in each row of `dst` are filled with 0. Inactive elements (positions outside the valid region) are also treated as 0.

Where `validRows = dst.GetValidRow()` and `validCols = dst.GetValidCol()`.

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = tpairreducesum %src : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tpairreducesum %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpairreducesum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename... WaitEvents>
PTO_INST RecordEvent TPAIRREDUCESUM(TileDataDst &dst, TileDataSrc0 &src0, WaitEvents &...events);
```

## Constraints

- **Implementation checks (A2A3 & A5)**:
    - `TileData::DType` must be `float` or `half`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
    - `dst` and `src0` must have the same `DType`.
    - `src0` must have the same valid shape as `dst` (`src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`).
- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.
    - The lower half of each destination row (positions `0 … ⌈validCols/2⌉−1`) holds the pair-reduced results; the upper half (positions `⌈validCols/2⌉ … validCols−1`) is filled with 0. Inactive elements outside the valid region are also treated as 0.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
    using TileDst = Tile<TileType::Vec, float, 16, 32>;
    using TileSrc = Tile<TileType::Vec, float, 16, 64>;
    TileDst dst(16, 32);
    TileSrc src0(16, 32);

    TPAIRREDUCESUM(dst, src0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
    using TileDst = Tile<TileType::Vec, half, 16, 64, BLayout::RowMajor, 16, 64>;
    using TileSrc = Tile<TileType::Vec, half, 16, 128, BLayout::RowMajor, 16, 128>;
    TileDst dst;
    TileSrc src0;

    TASSIGN(src0, 0x1000);
    TASSIGN(dst, 0x2000);

    TPAIRREDUCESUM(dst, src0);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tpairreducesum %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %src,  @tile(0x1000)
# pto.tassign %dst,  @tile(0x2000)
%dst = pto.tpairreducesum %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tpairreducesum %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.tpairreducesum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Instructions

- [TDeInterleave](TDEINTERLEAVE.md) - De-interleave splits even/odd positions; TPairReduceSum sums adjacent pairs.
- [TInterleave](TINTERLEAVE.md) - Interleave alternates elements from two sources into a combined stream.
