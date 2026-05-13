# pto.texpands

`pto.texpands` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TEXPANDS` broadcasts a scalar value into a destination tile.

## Semantics

For vector tiles, `TEXPANDS` fills the destination over its active valid region with the same scalar value.

For matrix / conv-tile paths on NPU backends, the implementation fills the full backend-defined matrix storage region rather than using the vector valid-region iteration model.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = texpands %scalar : dtype -> !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.texpands ins(%scalar : dtype)
            outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - NPU backends accept vector or matrix destination locations depending on the path.
    - Vector-tile paths use the destination valid region.
    - Matrix / conv-tile paths use backend-specific full-storage fill semantics.
    - Supported element types differ slightly between A2/A3 and A5.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | Vec path | valid-region fill | Supported | Supported |
    | Mat path | not documented in same backend terms | Supported | Supported |
    | `bf16` | not verified in this pass | Supported | No in verified vec-type list |
    | 8/16/32-bit ints | implementation-dependent | Supported | Supported |
    | `half` / `float` | Supported | Supported | Supported |

    Notes:

    - CPU simulator fills over `dst.GetValidRow()` / `dst.GetValidCol()` using the tile offset mapping.
    - A2/A3 and A5 both expose matrix/convtile fill paths with repeat-count limits based on backend matrix fill instructions.
    - A5 verified file supports `bfloat16_t` in its compile-time type check, but the older prose in the previous doc mixed vec/mat constraints imprecisely; this page keeps the statement backend-focused.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT dst;
    TEXPANDS(dst, 0.0f);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Next op in instruction set: [pto.tcmps](./tcmps.md)
