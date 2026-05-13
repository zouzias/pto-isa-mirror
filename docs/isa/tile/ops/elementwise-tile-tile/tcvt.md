# pto.tcvt

`pto.tcvt` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

`TCVT` performs lane-wise type conversion from a source tile into a destination tile, under an explicit rounding mode and optional saturation mode.

## Semantics

For each active lane in the conversion domain, `TCVT` converts the source element to the destination element type using the selected rounding policy and, where applicable, saturation policy.

The exact supported type pairs and edge-case handling differ by backend.

## Rounding Modes

Common documented rounding modes include:

- `RoundMode::CAST_RINT`
- `RoundMode::CAST_ROUND`
- `RoundMode::CAST_FLOOR`
- `RoundMode::CAST_CEIL`
- `RoundMode::CAST_TRUNC`
- `RoundMode::CAST_ODD` on backends that implement it

## Saturation Modes

When a `SaturationMode` parameter is provided, the backend uses the requested saturation policy where that conversion path supports it.

## Syntax

### Assembly Form (PTO-AS)

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```mlir
%dst = pto.tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```mlir
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

`TCVT` exposes overloads with and without an explicit temporary tile, and with and without explicit saturation mode.

## Constraints

!!! warning "Constraints"
    - `src` and `dst` must use a backend-supported conversion type pair.
    - Shape / valid-region compatibility is required for legal use.
    - Some backend paths require temporary storage and therefore use the `tmp` overloads.
    - Saturation and edge-case behavior are conversion-path dependent.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **CPU simulator**:
        - implements scalarized lane-wise conversion with explicit rounding helpers and optional saturation handling.
    - **A2/A3**:
        - supports a large but backend-specific set of conversion pairs including float/integer, half/integer, bf16-related paths, and selected int64 paths.
        - some non-saturating edge-aligned paths rely on temporary storage helpers.
    - **A5**:
        - supports a large backend-specific conversion matrix including integer, half, float, bf16, and fp8-related paths.
        - saturation control is implemented through backend control-bit configuration and path-specific helper selection.

    Notes:

    - The exact conversion matrix is substantial and implementation-specific; this page intentionally avoids overstating unsupported or unverified type-pair guarantees.
    - `CAST_ODD` is present in backend implementations and should be treated as backend-dependent rather than universally guaranteed for all type pairs.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
    using SrcT = Tile<TileType::Vec, float, 16, 16>;
    using DstT = Tile<TileType::Vec, half, 16, 16>;
    SrcT src;
    DstT dst;
    TCVT(dst, src, RoundMode::CAST_RINT);
}
```

```cpp
using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
TmpT tmp;
TCVT(dst, src, tmp, RoundMode::CAST_TRUNC, SaturationMode::OFF);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tsubc](./tsubc.md)
- Next op in instruction set: [pto.tsel](./tsel.md)
