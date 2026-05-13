# pto.tmaxs

`pto.tmaxs` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TMAXS` performs lane-wise maximum selection between a source tile and a scalar value.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

## Semantics

`TMAXS` reads the source tile over the destination valid region and compares each lane against the scalar value, writing the lane-wise maximum into `dst`.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tmaxs %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMAXS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - The scalar type must match the source tile element type.
    - NPU backends require vector-tile execution paths.
    - The documented backends require at least matching valid columns; A2/A3 also explicitly checks matching valid rows and non-zero destination valid extents.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `f32` | not verified in this pass | Supported | Supported |
    | `f16` | not verified in this pass | Supported | Supported |
    | `i32` | not verified in this pass | Supported | Supported |
    | `i16` | not verified in this pass | Supported | Supported |
    | `u32` | not verified in this pass | No | Supported |
    | `u16` | not verified in this pass | No | Supported |
    | `u8` | not verified in this pass | No | Supported |
    | `i8` | not verified in this pass | No | Supported |
    | `bf16` | not verified in this pass | No | No in verified file | Supported in doc was overstated |

    Notes:

    - A2/A3 currently accepts `int32_t`, `int16_t`, `half`/`float16_t`, and `float`/`float32_t`.
    - A5 currently accepts `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, `half`, `bfloat16_t`, `uint8_t`, and `int8_t`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT x, out;
    TMAXS(out, x, 0.0f);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.trems](./trems.md)
- Next op in instruction set: [pto.tands](./tands.md)
