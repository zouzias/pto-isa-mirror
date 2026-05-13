# pto.tsubs

`pto.tsubs` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TSUBS` performs lane-wise subtraction of a scalar from a source tile.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{scalar} $$

## Semantics

`TSUBS` reads the source tile over the destination valid region and subtracts the scalar value from each lane, writing the result into `dst`.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tsubs %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - The scalar type must match the source tile element type.
    - NPU backends require vector-tile execution paths.
    - The documented backends require matching valid columns and rows.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `f32` | Supported via CPU scalar-op framework | Supported | Supported |
    | `f16` | Supported via CPU scalar-op framework | Supported | Supported |
    | `i32` | Supported via CPU scalar-op framework | Supported | Supported |
    | `i16` | Supported via CPU scalar-op framework | Supported | Supported |
    | `u32` | Supported via CPU scalar-op framework | No | Supported |
    | `u16` | Supported via CPU scalar-op framework | No | Supported |
    | `u8` | Supported via CPU scalar-op framework | No | Supported |
    | `i8` | Supported via CPU scalar-op framework | No | Supported |
    | `bf16` | not verified in CPU pass | No | Supported |

    Notes:

    - A2/A3 currently accepts `int32_t`, `int16_t`, `half`/`float16_t`, and `float`/`float32_t`.
    - A5 currently accepts `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `uint16_t`, `uint8_t`, `half`, `float32_t`, and `bfloat16_t`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT x, out;
    TSUBS(out, x, 1.0f);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tadds](./tadds.md)
- Next op in instruction set: [pto.tdivs](./tdivs.md)
