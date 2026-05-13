# pto.tand

`pto.tand` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TAND` performs lane-wise bitwise AND of two source tiles into a destination tile.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \;\&\; \mathrm{src1}_{i,j} $$

## Semantics

`TAND` reads corresponding lanes from `src0` and `src1` over the destination valid region and writes the bitwise-AND result into `dst`.

Backend implementations require the source valid regions to match the destination valid region.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tand %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tand %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tand ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst`, `src0`, and `src1` must use the same element type.
    - `src0`, `src1`, and `dst` must have the same runtime valid shape.
    - A2/A3 and A5 require RowMajor execution paths.
    - Bitwise support is constrained by backend element-width support.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `u8` | Not documented | Supported | Supported |
    | `i8` | Not documented | Supported | Supported |
    | `u16` | Not documented | Supported | Supported |
    | `i16` | Not documented | Supported | Supported |
    | `u32` | Not documented | No | Supported |
    | `i32` | Not documented | No | Supported |

    Notes:

    - A2/A3 accepts 1-byte and 2-byte integral element widths.
    - A5 accepts 1-byte, 2-byte, and 4-byte integral element widths.
    - This page stays conservative for CPU simulator support because no dedicated CPU `TAND` implementation was verified in this pass.

## Performance

### A2/A3 Throughput

`TAND` is a binary vector op using the same A2/A3 timing family as other binary arithmetic/logical instructions.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
    TileT a, b, out;
    TAND(out, a, b);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tabs](./tabs.md)
- Next op in instruction set: [pto.tor](./tor.md)
