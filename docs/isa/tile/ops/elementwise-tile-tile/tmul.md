# pto.tmul

`pto.tmul` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TMUL` performs lane-wise multiplication of two source tiles into a destination tile.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j} $$

## Semantics

`TMUL` reads corresponding lanes from `src0` and `src1` over the destination valid region and writes the product into `dst`.

Backend implementations require the source valid regions to match the destination valid region.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tmul %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst`, `src0`, and `src1` must use the same element type.
    - `src0`, `src1`, and `dst` must have the same runtime valid shape.
    - A2/A3 and A5 require RowMajor vector-tile execution paths.
    - CPU simulator supports both RowMajor and non-RowMajor layouts implemented by tile offset mapping.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `f32` | Supported | Supported | Supported |
    | `f16` | Supported | Supported | Supported |
    | `i32` | Supported | Supported | Supported |
    | `i16` | Supported | Supported | Supported |
    | `u32` | Supported | No | Supported |
    | `u16` | Supported | No | Supported |

    Notes:

    - A2/A3 currently accepts `int32_t`, `int16_t`, `half`, and `float`.
    - A5 currently accepts `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, and `half`.

## Performance

### A2/A3 Throughput

`TMUL` is a binary vector op using the same A2/A3 timing family as other binary arithmetic instructions, with multiplication-specific completion latency.

| Metric | FP | INT |
|---|---|---|
| Startup latency | 14 | 14 |
| Completion latency | 20 | 18 |
| Per-repeat throughput | 2 | 2 |
| Pipeline interval | 18 | 18 |

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src0, src1, dst;
    TMUL(dst, src0, src1);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tsub](./tsub.md)
- Next op in instruction set: [pto.tmin](./tmin.md)
