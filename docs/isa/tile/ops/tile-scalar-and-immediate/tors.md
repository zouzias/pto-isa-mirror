# pto.tors

`pto.tors` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TORS` performs lane-wise bitwise OR between a source tile and a scalar value.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;|\; \mathrm{scalar} $$

## Semantics

The NPU backends implement `TORS` by first materializing the scalar into the destination tile and then applying tile-tile `TOR` between `src` and that expanded destination tile.

This means destination aliasing constraints matter in manual mode on some backends.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tors %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
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
    | Scalar-op framework | Supported | implemented via `TEXPANDS + TOR` | implemented via `TEXPANDS + TOR` |
    | Manual-mode `dst == src` alias | not documented the same way | Unsupported | unsupported by construction / should be avoided |
    | Element-type support | follows CPU scalar-op framework | integral element types compatible with `TEXPANDS` + `TOR` | integral element types compatible with `TEXPANDS` + `TOR` |

    Notes:

    - A2/A3 explicitly rejects `dst` and `src` bound to the same memory in manual mode.
    - A5 implementation is thin and relies on `TEXPANDS` plus `TOR`, so the same conceptual alias hazard applies.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
    TileT dst, src;
    TORS(dst, src, 0xffu);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tands](./tands.md)
- Next op in instruction set: [pto.tshls](./tshls.md)
