# pto.tcmps

`pto.tcmps` is part of the [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md) instruction set.

## Summary

`TCMPS` compares a source tile against either a scalar value or a scalar carrier tile and writes a packed predicate-result tile.

## Semantics

For each logical source lane in the active comparison domain, `TCMPS` evaluates:

$$ \mathrm{src}_{i,j}\ \mathrm{cmpMode}\ \mathrm{rhs} $$

where `rhs` is either:

- a scalar immediate / scalar value passed directly, or
- a tile operand whose first element is used as the scalar comparison value by the backend.

The destination stores a backend-defined packed predicate encoding rather than a normal arithmetic tile payload.

Supported compare modes are:

- `CmpMode::EQ`
- `CmpMode::NE`
- `CmpMode::LT`
- `CmpMode::LE`
- `CmpMode::GT`
- `CmpMode::GE`

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tcmps %src, %scalar {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tcmps %src, %scalar {cmpMode = #pto.cmp<EQ>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tcmps ins(%src, %scalar {cmpMode = #pto.cmp<EQ>} : !pto.tile_buf<...>, dtype)
          outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1,
                           CmpMode mode, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1,
                           CmpMode mode, WaitEvents &... events);
```

## Constraints

!!! warning "Constraints"
    - Destination tile must be a packed predicate-result tile compatible with the selected backend.
    - Source tile must be a vector tile on NPU backends.
    - Backends require RowMajor destination layout.
    - The documented backends explicitly require `src0.GetValidRow() == dst.GetValidRow()`.
    - For the tile-RHS form, the RHS tile element type must match the source tile element type.
    - The packed destination encoding is target-specific and must not be interpreted as a normal arithmetic tile payload.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Property | CPU Simulator | A2/A3 | A5 |
    |---|---|---|---|
    | Supported source types | current CPU implementation focuses on scalar/tile compare emulation | `int32_t`, `float`, `half`, `uint16_t`, `int16_t` | `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, `half`, `uint8_t`, `int8_t` |
    | Destination kind | packed result bytes in current CPU implementation | packed predicate result tile | packed predicate result tile |
    | Tile-RHS form | supported in API surface | supported | supported |
    | Layout requirement | implementation-defined CPU layout handling | RowMajor dst | RowMajor dst |

    Notes:

    - On A2/A3, `int32_t` compare currently dispatches through EQ-only scalar compare behavior.
    - On A2/A3 and A5, the tile-RHS overload reads the RHS scalar value from the provided tile payload.
    - CPU simulator packs results into byte-oriented storage in its current implementation.

## Relationship with `TSEL`

`TCMPS` commonly produces a packed predicate tile that can later be consumed by `TSEL` / `TSELS`, but producer and consumer must agree on the same backend-specific packing convention.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using SrcT = Tile<TileType::Vec, float, 16, 16>;
    using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
    SrcT src;
    DstT dst(16, 2);
    TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Tile Scalar And Immediate](../../tile-scalar-and-immediate.md)
- Previous op in instruction set: [pto.texpands](./texpands.md)
- Next op in instruction set: [pto.tsels](./tsels.md)
