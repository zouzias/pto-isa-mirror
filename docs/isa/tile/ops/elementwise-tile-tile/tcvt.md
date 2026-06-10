# pto.tcvt

`pto.tcvt` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise type conversion with a specified rounding mode and optional saturation mode.

## Mechanism

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode},\mathrm{satmode}}\!\left(\mathrm{src}_{i,j}\right) $$

where `rmode` is the rounding policy and `satmode` (if provided) controls saturation behavior.

## Rounding Modes

| Mode | Behavior |
|------|----------|
| `RoundMode::CAST_RINT` | Round to nearest, ties to even |
| `RoundMode::CAST_ROUND` | Round to nearest, ties away from zero |
| `RoundMode::CAST_FLOOR` | Round toward -∞ |
| `RoundMode::CAST_CEIL` | Round toward +∞ |
| `RoundMode::CAST_TRUNC` | Round toward zero |

## Saturation Modes

When `SaturationMode` is provided, saturation behavior is explicitly controlled:

| Mode | Behavior |
|------|----------|
| `SaturationMode::ON` | Saturation enabled |
| `SaturationMode::OFF` | Saturation disabled |

When `SaturationMode` is omitted, the implementation chooses the default behavior for the selected target/type path. Some conversion paths also expose a `tmp`-tile overload used for explicit scratch storage.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);
```

The `tmp`-tile overloads exist for conversion paths that need explicit scratch storage.

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src` | Source tile | Source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%dst` | Destination tile | Destination tile receiving the converted values |
| `mode` | Rounding mode | One of `CAST_RINT`, `CAST_ROUND`, `CAST_FLOOR`, `CAST_CEIL`, `CAST_TRUNC` |
| `satMode` | Saturation mode (optional) | `ON` or `OFF` |
| `tmp` | Temporary tile (optional) | Scratch tile for conversion paths that require explicit temporary storage |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain the converted element values after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - `src` and `dst` MUST have compatible shapes (declared shape and valid region).
    - The source/destination type pair MUST be supported by the selected target profile.
    - The rounding mode MUST be supported for the given type pair.
    - When a conversion path requires explicit scratch storage, callers MUST use one of the `tmp`-tile overloads.
    - Disabling saturation may change overflow behavior for some backend/type paths, especially low-precision integer conversions.

## Cases That Are Not Allowed

!!! danger "Cases That Are Not Allowed"
    - **MUST NOT** use a type pair not supported by the target profile.
    - **MUST NOT** use a rounding mode not supported for the given type pair.
    - **MUST NOT** assume that disabling saturation still clamps overflow to the destination range.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    `pto.tcvt` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but the exact set of supported type pairs, scratch requirements, and saturation behavior is backend-specific.

    In this checkout, the fp16 → int8 non-saturating path is explicitly implemented through helper logic that may require temporary storage and row-aware sub-chunking.

## Supported Conversions

| Source Type | A2A3 Destinations | A5 Destinations | Difference |
|---|---|---|---|
| FP32 | FP16, FP32 (round-only), BF16, I16, I32, I64 | FP32, FP16, BF16, I16, I32, I64, FP8_E4M3, FP8_E5M2, H8 | A5 adds FP8/H8 targets |
| FP16 | FP32, I32, I16, I8, U8, S4 (int4b_t) | FP32, I32, I16, I8, U8, H8 | A2A3 has S4 path; A5 has H8 path |
| BF16 | FP32, I32 | FP32, I32, FP16, FP4_E1M2X2, FP4_E2M1X2 | A5 adds FP16/FP4 targets |
| I16 | FP16, FP32 | U8, FP16, FP32, U32, I32 | A5 adds U8/U32/I32 targets |
| I32 | FP32, I16, I64, FP16 (deq path) | FP32, I16, U16, I64, U8 | A2A3 supports I32 -> FP16 (half, deq); A5 does not |
| I64 | FP32, I32 | FP32, I32 | Same |
| U8 | FP16 | FP16, U16 | A5 adds U16 target |
| I8 | FP16 | FP16, I16, I32 | A5 adds I16/I32 targets |
| S4 (int4b_t) | FP16 | N/A | A2A3-only |
| U32 | N/A | U8, U16, I16 | A5-only source type |
| FP8_E4M3 | N/A | FP32 | A5-only source type |
| FP8_E5M2 | N/A | FP32 | A5-only source type |
| H8 | N/A | FP32 | A5-only source type |
| FP4_E1M2X2 | N/A | BF16 | A5-only source type |
| FP4_E2M1X2 | N/A | BF16 | A5-only source type |

Notes:

- A2A3 supports I32 -> FP16 through the half dequantization path; A5 does not support I32 -> FP16.
- A5 does not support FP16 -> FP8_E4M3 or FP16 -> FP8_E5M2.

## Examples

### Auto

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

### Explicit Saturation / Scratch

```cpp
using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
TmpT tmp;
TCVT(dst, src, tmp, RoundMode::CAST_TRUNC, SaturationMode::OFF);
```

### PTO Assembly Form

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tsubc](./tsubc.md)
- Next op in instruction set: [pto.tsel](./tsel.md)
