# pto.tsel

`pto.tsel` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TSEL` performs lane-wise conditional selection between two source tiles under control of a packed predicate mask tile.

## Semantics

For each logical destination lane in the active valid region, `TSEL` selects either `src0` or `src1` according to the predicate information encoded in `selMask`.

The exact in-memory representation of `selMask` is backend-specific and is tied to the packed compare-mask conventions used by predicate-producing operations such as `TCMP` / `TCMPS`.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0,
                          TileData &src1, TmpTile &tmp, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst`, `src0`, and `src1` must use the same element type.
    - A2/A3 requires 2-byte or 4-byte destination/source element sizes.
    - A5 supports 1-byte, 2-byte, and 4-byte destination/source element sizes.
    - `dst`, `src0`, and `src1` must use RowMajor layout on NPU backends.
    - `selMask` must use a predicate-mask storage format compatible with the selected backend.
    - `tmp` is required by the API and participates in backend-specific mask handling.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | Data element sizes | follows CPU implementation behavior | 2 or 4 bytes | 1, 2, or 4 bytes |
    | RowMajor requirement | not enforced in the same way as NPU docs | Required | Required |
    | Packed predicate mask | byte-packed CPU interpretation | backend-specific packed format | backend-specific packed format |
    | `tmp` required in API | Yes | Yes | Yes |

    Notes:

    - CPU simulator currently interprets the mask tile as packed bits and does not use `tmp` for functional behavior.
    - A2/A3 uses temporary storage to materialize compare masks before issuing `vsel`.
    - A5 has separate 32-bit and 8/16-bit selection paths and still requires the `tmp` parameter in the public API.

## Relationship with `TCMP` / `TCMPS`

`TSEL` is typically paired with a packed predicate tile produced by `TCMP` or `TCMPS`. The producer and consumer must agree on the same backend-specific packing convention.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
    using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
    TileT src0, src1, dst;
    MaskT mask(16, 2);
    TmpT tmp;
    TSEL(dst, mask, src0, src1, tmp);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tcvt](./tcvt.md)
- Next op in instruction set: [pto.trsqrt](./trsqrt.md)
