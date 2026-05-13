# pto.tabs

`pto.tabs` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TABS` computes the elementwise absolute value of a source tile into a destination tile.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right| $$

## Semantics

`TABS` iterates over the destination valid region and applies absolute value lane-wise.

Backend implementations expect the source and destination valid shapes to match at use sites.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tabs %src : !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tabs ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `src` and `dst` must use a backend-supported element type.
    - A2/A3 and A5 support RowMajor vector-tile execution paths.
    - CPU simulator supports the documented CPU implementation types and uses tile offset mapping.
    - Source and destination valid shapes should match for legal use.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `f32` | Supported | Supported | Supported |
    | `f16` | Supported | Supported | Supported |
    | `bf16` | Supported | No | Supported |
    | `i32` | Supported | No | No |
    | `i16` | Supported | No | No |

    Notes:

    - CPU simulator currently accepts `int32_t`/`int`, `int16_t`, `half`, `bfloat16_t`, and `float`.
    - NPU `TABS` is implemented through unary vector paths and is narrower in type support than binary ops.

## Performance

### A2/A3 Throughput

`TABS` is implemented through the unary-op backend path. Exact total cycles depend on the selected unary instruction schedule and tile geometry.

## Examples

### C++

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src, dst;
    TABS(dst, src);
}
```

### MLIR

```mlir
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tadd](./tadd.md)
- Next op in instruction set: [pto.tand](./tand.md)
