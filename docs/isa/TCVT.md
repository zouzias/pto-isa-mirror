# TCVT

## Introduction

Elementwise type conversion with a specified rounding mode.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$
\mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode}}\!\left(\mathrm{src}_{i,j}\right)
$$

where `rmode` is a rounding policy (see `pto::RoundMode`).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, RoundMode mode, WaitEvents&... events);
```

## Constraints

- `dst` and `src` must be compatible in shape/valid region as required by the implementation.
- The conversion `(src element type) -> (dst element type)` must be supported by the target for the given `RoundMode`.
- **Implementation notes (A2A3/A5)**:
  - `TCVT_IMPL` does not enforce additional `static_assert`/`PTO_ASSERT` checks on the type pair; unsupported conversions are target-defined.

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

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

## Architecture Comparison: A5 vs A2/A3

The TCVT instruction has different implementation capabilities across NPU architectures. This section provides a comprehensive comparison.

### Implementation Files

- **A5 Architecture**: `include/pto/npu/a5/TCvt.hpp` - Conversion support with FP8 formats
- **A2/A3 Architecture**: `include/pto/npu/a2a3/TCvt.hpp` - Core conversion support for standard types

### Complete Conversion Support Matrix

| Source Type            | Destination Type              | A5 | A2/A3 |
| ---------------------- | ----------------------------- | -- | ----- |
| **FP32 (float)** | → float (same-type rounding) | ✅ | ✅    |
|                        | → half (FP16)                | ✅ | ✅    |
|                        | → bfloat16                   | ✅ | ✅    |
|                        | → int16                      | ✅ | ✅    |
|                        | → int32                      | ✅ | ✅    |
|                        | → int64                      | ✅ | ✅    |
|                        | → float8_e4m3                | ✅ | ❌    |
|                        | → float8_e5m2                | ✅ | ❌    |
|                        | → hifloat8                   | ✅ | ❌    |
| **FP16 (half)**  | → float                      | ✅ | ✅    |
|                        | → int32                      | ✅ | ✅    |
|                        | → int16                      | ✅ | ✅    |
|                        | → int8                       | ✅ | ✅    |
|                        | → uint8                      | ✅ | ✅    |
|                        | → hifloat8                   | ✅ | ❌    |
| **BFloat16**     | → float                      | ✅ | ✅    |
|                        | → int32                      | ✅ | ✅    |
|                        | → half                       | ✅ | ❌    |
| **uint8**        | → half                       | ✅ | ✅    |
|                        | → uint16                     | ✅ | ❌    |
| **int8**         | → half                       | ✅ | ✅    |
|                        | → int16                      | ✅ | ❌    |
|                        | → int32                      | ✅ | ❌    |
| **int16**        | → uint8                      | ✅ | ❌    |
|                        | → half                       | ✅ | ✅    |
|                        | → float                      | ✅ | ✅    |
|                        | → uint32                     | ✅ | ❌    |
|                        | → int32                      | ✅ | ❌    |
| **int32**        | → float                      | ✅ | ✅    |
|                        | → int16                      | ✅ | ✅    |
|                        | → uint16                     | ✅ | ❌    |
|                        | → int64                      | ✅ | ✅    |
|                        | → uint8                      | ✅ | ❌    |
| **uint32**       | → uint8                      | ✅ | ❌    |
|                        | → uint16                     | ✅ | ❌    |
|                        | → int16                      | ✅ | ❌    |
| **int64**        | → float                      | ✅ | ✅    |
|                        | → int32                      | ✅ | ✅    |
| **float8_e4m3**  | → float                      | ✅ | ❌    |
| **float8_e5m2**  | → float                      | ✅ | ❌    |
| **hifloat8**     | → float                      | ✅ | ❌    |

### Architecture-Specific Features

Both architectures support comprehensive rounding control:

| Mode                              | Description               |            |
| --------------------------------- | ------------------------- | ---------- |
| **Round to Nearest (Even)** | IEEE 754 default rounding | CAST_RINT  |
| **Round Away from Zero**    | Arithmetic rounding       | CAST_ROUND |
| **Round Down (Floor)**      | Toward negative infinity  | CAST_FLOOR |
| **Round Up (Ceiling)**      | Toward positive infinity  | CAST_CEIL  |
| **Truncate (toward zero)**  | Discard fractional part   | CAST_TRUNC |
| **Odd Rounding**            | Round to odd (FP32 only)  | CAST_ODD   |
