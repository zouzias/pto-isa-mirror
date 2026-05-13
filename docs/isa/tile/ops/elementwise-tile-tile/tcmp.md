# pto.tcmp

`pto.tcmp` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TCMP` compares two source tiles lane-wise and writes a packed predicate-result tile.

The destination tile stores a backend-defined packed predicate encoding rather than one scalar boolean object per logical lane.

## Semantics

For each logical source lane in the active comparison domain, `TCMP` evaluates:

$$ \mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j} $$

and stores the comparison result into the packed predicate destination tile.

Supported compare modes are:

- `CmpMode::EQ`
- `CmpMode::NE`
- `CmpMode::LT`
- `CmpMode::LE`
- `CmpMode::GT`
- `CmpMode::GE`

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tcmp ins(%src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1,
                          CmpMode cmpMode, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - Source tiles must be vector tiles.
    - Destination tile must be a vector tile carrying the packed predicate result.
    - Backends require target-compatible predicate destination layout and storage shape.
    - A2/A3 explicitly checks `src0.GetValidRow() == src1.GetValidRow()`, `src0.GetValidCol() == src1.GetValidCol()`, and `src0.GetValidRow() == dst.GetValidRow()`.
    - The packed destination encoding is target-specific and must not be interpreted as a normal arithmetic tile payload.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | A2/A3 | A5 |
    |---|---|---|
    | Supported source types | `int32_t`, `half`, `float` | `uint32_t`, `int32_t`, `uint16_t`, `int16_t`, `uint8_t`, `int8_t`, `float`, `half` |
    | Source tile location | `TileType::Vec` | effectively vector-path compare |
    | Predicate result storage | packed compare-result tile | packed compare-result tile |
    | Compare modes | `int32_t` path effectively uses EQ-only; other listed types support mode dispatch | listed types support mode dispatch |

    Notes:

    - A2/A3 `int32_t` compare currently dispatches through the EQ compare path regardless of `cmpMode`.
    - A5 stores packed predicate results through predicate-store helpers and uses a different packing scheme from A2/A3.
    - This page does not claim CPU simulator support because no dedicated CPU `TCMP` implementation was verified in this pass.

## Relationship with `TSEL`

`TCMP` commonly produces a packed predicate tile that can later be consumed by `TSEL` or related predicate-selection workflows, but consumers must respect the backend-specific packing contract.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using SrcT = Tile<TileType::Vec, float, 16, 16>;
    using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
    SrcT src0, src1;
    MaskT mask(16, 2);
    TCMP(mask, src0, src1, CmpMode::GT);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tmax](./tmax.md)
- Next op in instruction set: [pto.tdiv](./tdiv.md)
