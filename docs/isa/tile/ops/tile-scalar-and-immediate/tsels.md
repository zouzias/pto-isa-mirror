# pto.tsels

`pto.tsels` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TSELS` performs lane-wise conditional selection between a source tile and a scalar fallback value under control of a packed predicate mask tile.

## Semantics

For each logical destination lane in the active valid region, `TSELS` selects either:

- the corresponding `src` lane, or
- the scalar fallback value,

according to the predicate information encoded in `mask`.

The exact in-memory representation of `mask` is backend-specific and tied to the packed compare-mask conventions used by predicate-producing operations such as `TCMP` / `TCMPS`.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tsels %mask, %src, %scalar : !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tsels %mask, %src, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tsels ins(%mask, %src, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype)
          outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp,
                           typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - `dst` and `src` must use the same element type.
    - A2/A3 supports 2-byte and 4-byte data-element sizes for `dst/src`.
    - A5 supports 1-byte, 2-byte, and 4-byte data-element sizes for `dst/src`.
    - A2/A3 requires RowMajor `dst` and `src`.
    - A5 requires RowMajor `dst`, `mask`, and `src`.
    - `src.GetValidRow()/GetValidCol()` must match `dst.GetValidRow()/GetValidCol()` on the documented NPU backends.
    - `mask` must use a backend-compatible packed predicate format.
    - `tmp` is required by the public API and participates in backend-specific scalar/mask handling.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | Selection model | current CPU implementation differs from NPU mask-based contract | packed-mask tile + scalar fallback | packed-mask tile + scalar fallback |
    | Supported `dst/src` types | not aligned with NPU API contract | `half` / `float16_t`, `float` / `float32_t` | `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `float` |
    | Element sizes | implementation-defined | 2 or 4 bytes | 1, 2, or 4 bytes |
    | `tmp` required in API | Yes | Yes | Yes |

    Notes:

    - A2/A3 materializes scalar/mask state through the `tmp` tile and vector compare-mask path.
    - A5 has separate 32-bit and 8/16-bit selection paths.
    - The current CPU simulator implementation named `TSELS_IMPL` does not match the NPU mask-plus-scalar contract exactly, so this page keeps CPU behavior conservative and NPU-focused.

## Relationship with `TCMPS`

`TSELS` is typically paired with a packed predicate tile produced by `TCMPS`. The producer and consumer must agree on the same backend-specific packing convention.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using TileDst = Tile<TileType::Vec, float, 16, 16>;
    using TileSrc = Tile<TileType::Vec, float, 16, 16>;
    using TileTmp = Tile<TileType::Vec, float, 16, 16>;
    using TileMask = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
    TileDst dst;
    TileSrc src;
    TileTmp tmp;
    TileMask mask(16, 2);
    float scalar = 0.0f;
    TSELS(dst, mask, src, tmp, scalar);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.tcmps](./tcmps.md)
- Next op in instruction set: [pto.tmins](./tmins.md)
