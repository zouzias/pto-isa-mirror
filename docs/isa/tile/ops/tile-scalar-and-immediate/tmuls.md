# pto.tmuls

`pto.tmuls` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TMULS` performs lane-wise multiplication between a source tile and a scalar value.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \cdot \mathrm{scalar} $$

## Semantics

`TMULS` reads the source tile over the destination valid region and multiplies each lane by the scalar value, writing the result into `dst`.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tmuls %src, %scalar : !pto.tile<...>, dtype
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tmuls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tmuls ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMULS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - The scalar type must match the source tile element type.
    - NPU backends require vector-tile execution paths.
    - The documented backends require at least matching valid columns; A2/A3 also explicitly checks matching valid rows.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `f32` | Supported via CPU scalar-op framework | Supported | Supported |
    | `f16` | Supported via CPU scalar-op framework | Supported | Supported |
    | `i32` | Supported via CPU scalar-op framework | Supported | Supported |
    | `i16` | Supported via CPU scalar-op framework | Supported | Supported |
    | `u32` | not verified in CPU pass | No | No in verified file |
    | `u16` | not verified in CPU pass | No | No in verified file |
    | `u8` | not verified in CPU pass | No | No in verified file |
    | `i8` | not verified in CPU pass | No | No in verified file |
    | `bf16` | not verified in CPU pass | No | Supported |

    Notes:

    - A2/A3 currently accepts `int32_t`, `int16_t`, `half`/`float16_t`, and `float`/`float32_t`.
    - A5 verified file accepts `int32_t`, `int16_t`, `half`/`float16_t`, `float`/`float32_t`, and `bfloat16_t`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src, dst;
    TMULS(dst, src, 2.0f);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tdivs](./tdivs.md)
- Next op in instruction set: [pto.tfmods](./tfmods.md)
