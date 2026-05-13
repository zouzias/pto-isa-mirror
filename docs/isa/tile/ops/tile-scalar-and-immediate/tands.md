# pto.tands

`pto.tands` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TANDS` performs lane-wise bitwise AND between a source tile and a scalar value.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;\&\; \mathrm{scalar} $$

## Semantics

The NPU backends implement `TANDS` by first materializing the scalar into the destination tile and then applying tile-tile `TAND` between `src` and that expanded destination tile.

This means destination aliasing constraints matter in manual mode on some backends.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tands %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TANDS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - The scalar type must match the tile element type.
    - NPU backends require vector-tile execution paths.
    - Because the scalar is expanded into `dst`, manual-mode aliasing between `dst` and `src` is unsupported on documented NPU implementations.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | Scalar-op framework | Supported | implemented via `TEXPANDS + TAND` | implemented via `TEXPANDS + TAND` |
    | Manual-mode `dst == src` alias | not documented the same way | Unsupported | unsupported by construction / should be avoided |
    | Element-type support | follows CPU scalar-op framework | integral element types compatible with `TEXPANDS` + `TAND` | integral element types compatible with `TEXPANDS` + `TAND` |

    Notes:

    - A2/A3 explicitly rejects `dst` and `src` bound to the same memory in manual mode.
    - A5 implementation is thin and relies on `TEXPANDS` plus `TAND`, so the same conceptual alias hazard applies.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
    TileT dst, src;
    TANDS(dst, src, 0xffu);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tmaxs](./tmaxs.md)
- Next op in instruction set: [pto.tors](./tors.md)
