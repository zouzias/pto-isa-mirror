# pto.tdiv

`pto.tdiv` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TDIV` performs lane-wise division of two source tiles into a destination tile.

For every element inside the destination tile valid region:

$$ \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}} $$

## Semantics

`TDIV` reads corresponding lanes from `src0` and `src1` over the destination valid region and writes the quotient into `dst`.

Backend implementations require the source valid regions to match the destination valid region.

`PrecisionType` selects the backend division path where supported.

## Precision Modes

- `DivAlgorithm::DEFAULT`
- `DivAlgorithm::HIGH_PRECISION`

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tdiv %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst`, `src0`, and `src1` must use the same element type on the documented NPU backends.
    - `src0`, `src1`, and `dst` must have the same runtime valid shape on the documented NPU backends.
    - A2/A3 and A5 require RowMajor NPU execution paths.
    - Division-by-zero behavior is backend-defined.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Element type | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | `f32` | Supported | Supported | Supported |
    | `f16` | Supported | Supported | Supported |
    | `i32` | Supported | No | Supported |
    | `u32` | Supported | No | Supported |
    | `i16` | Supported | No | Supported |
    | `u16` | Supported | No | Supported |
    | `u8` | Supported | No | Supported |
    | `i8` | Supported | No | Supported |
    | `bf16` | not verified in CPU table | No | Supported |

    Notes:

    - A2/A3 currently accepts only `half`/`float16_t` and `float`/`float32_t`.
    - A5 currently accepts `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, `half`, `bfloat16_t`, `uint8_t`, and `int8_t`.
    - `DivAlgorithm::HIGH_PRECISION` is meaningfully implemented on A5 for selected floating-point paths; A2/A3 does not expose a distinct high-precision backend path.

## Performance

### A2/A3 Throughput

`TDIV` lowers through the binary-op helper family but has operation-specific backend cost characteristics; it should not be documented as identical to plain add/sub timing without a verified cost-model source.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src0, src1, dst;
    TDIV(dst, src0, src1);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tcmp](./tcmp.md)
- Next op in instruction set: [pto.tshl](./tshl.md)
