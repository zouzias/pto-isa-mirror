# pto.tshrs

`pto.tshrs` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TSHRS` performs lane-wise right shift of a source tile by a scalar shift amount.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \gg \mathrm{scalar} $$

## Semantics

`TSHRS` reads the source tile over the destination valid region and applies a scalar right-shift amount to each lane.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tshrs %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tshrs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHRS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - Supported element types are integral vector-tile types only.
    - The documented backends require matching valid columns and rows.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `i32` | Supported via CPU scalar-op framework | Supported | Supported |
    | `i16` | Supported via CPU scalar-op framework | Supported | Supported |
    | `u32` | Supported via CPU scalar-op framework | Supported | Supported |
    | `u16` | Supported via CPU scalar-op framework | Supported | Supported |
    | `i8` | not verified in CPU pass | No | Supported |
    | `u8` | not verified in CPU pass | No | Supported |

    Notes:

    - A2/A3 verified helper supports `int32_t`, `int16_t`, `uint32_t`, and `uint16_t` families.
    - A5 verified helper supports `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `uint16_t`, and `uint8_t`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
    TileT dst, src;
    TSHRS(dst, src, 0x2);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tshls](./tshls.md)
- Next op in instruction set: [pto.txors](./txors.md)
