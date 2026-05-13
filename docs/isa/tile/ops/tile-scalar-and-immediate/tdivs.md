# pto.tdivs

`pto.tdivs` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TDIVS` performs scalar-division forms between a source tile and a scalar value.

Supported public forms are:

- tile/scalar: `src / scalar`
- scalar/tile: `scalar / src`

## Semantics

For each active lane in the destination valid region:

- tile/scalar form computes:

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}} $$

- scalar/tile form computes:

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}} $$

Backend implementations differ for integer vs floating-point element types, and A5 exposes a high-precision path for selected floating-point cases.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tdivs %src, %scalar : !pto.tile<...>, dtype
%dst = tdivs %scalar, %src : dtype, !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0,
                           WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - NPU backends require vector-tile execution paths.
    - The documented backends require matching valid columns and rows.
    - Division-by-zero behavior is backend-defined; some CPU/NPU helper paths assert or produce target-specific IEEE/integer outcomes.

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
    | `u8` | not verified in CPU pass | No | Supported |
    | `i8` | not verified in CPU pass | No | Supported |
    | `bf16` | not verified in CPU pass | No | No in verified file |

    Notes:

    - A2/A3 verified file supports `int32_t`, `int16_t`, `half`/`float16_t`, and `float`/`float32_t`.
    - A5 verified file supports integer and floating-point families including `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, and `float`.
    - `DivAlgorithm::HIGH_PRECISION` is meaningful on A5 for selected floating-point paths.

## Related Op Note

There is no separate `trdivs.md` page in this checkout; the scalar/tile form is documented here as the second `TDIVS` overload.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src, dst;
    TDIVS(dst, src, 2.0f);
    TDIVS(dst, 2.0f, src);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tsubs](./tsubs.md)
- Next op in instruction set: [pto.tmuls](./tmuls.md)
