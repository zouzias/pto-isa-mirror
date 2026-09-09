# TDEQUANT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T03:54:17.971Z pushedAt=2026-08-29T09:05:18.427Z -->

## Introduction

Dequantizes a low-precision quantized tile (`S8`/`S16`) into a high-precision `FP32` tile. Performs **affine dequantization** on each element, which is the inverse operation of the integer quantization (INT8 symmetric/asymmetric) performed by `TQUANT`:

$$ \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} - \mathrm{offset}_{i}) \cdot \mathrm{scale}_{i} $$

`scale` and `offset` are FP32 parameters provided **per row** (broadcast along columns to the entire row), so the complete dequantization of "offset removal + inverse scaling" can be accomplished in a single instruction.

## Mathematical Semantics

Conceptually, for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \bigl(\mathrm{src}_{i,j} - \mathrm{offset}_{i}\bigr) \cdot \mathrm{scale}_{i} $$

- `src`: quantized integer code (`S8` or `S16`).
- `scale`, `offset`: per-row FP32 dequantization parameters (the parameter group is selected by row index `i` and broadcast along columns to the entire row); the valid column count of `scale` is `paraCols = max(1, scale.GetValidCol())`, and the parameter column subscript `paraCol = min(j, paraCols - 1)` is used only to clamp the read column subscript when the parameter tile has multiple columns (typically one scalar per row, `paraCols = 1`); the parameter group itself is determined by row `i`.
- It is the inverse of the integer affine quantization performed by `TQUANT`: in `TQUANT`, $q = \mathrm{round}(x / \mathrm{scale}) + \mathrm{offset}$, hence $x = (q - \mathrm{offset}) \cdot \mathrm{scale}$.

> Unless otherwise specified, the semantics are defined within the valid region, and target-dependent behavior is marked as implementation-defined. Both `scale` and `offset` are ISA-visible tile operands (not compiler scratch).

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`.
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &...events);
```

| Parameter | Direction | Meaning |
|------|------|------|
| `dst` | Output | Dequantization result tile, `FP32`, row-major |
| `src` | Input | Quantization source tile, `S8` or `S16`, row-major, with the same valid shape as `dst` |
| `scale` | Input | Per-row de-scaling coefficient, `FP32`, broadcast along columns |
| `offset` | Input | Per-row zero-point offset, `FP32`, broadcast along columns |
| `events...` | Input | Wait events (`WaitEvents`), implicit `TSYNC` before the instruction |

`scale` and `offset` must be of the same type (`TileDataPara`), and their dtypes must be consistent with `dst` (both `FP32`).

## Tile Size and Data Types

For the valid shape $M \times N$ of the source/destination tiles:

| Tile | Dtype | Valid Shape | Layout | Description |
|------|-------|---------|------|------|
| `dst` | `float32_t` | $M \times N$ | RowMajor | Dequantization result |
| `src` | `int8_t` or `int16_t` | $M \times N$ | RowMajor | Quantization integer code; the dtype determines the unpack path |
| `scale` | `float32_t` | $M \times 1$ (per row) | ColMajor/row broadcast | Broadcast along columns (`BRC_B32`) |
| `offset` | `float32_t` | $M \times 1$ (per row) | ColMajor/row broadcast | Broadcast along columns (`BRC_B32`) |

> The valid row count of `scale`/`offset` must equal the valid row count of `dst`; the column direction is broadcast in 32-byte blocks, so the typical usage is one scalar per row (shape $M \times 1$).

## Supported Input Dtypes

| Source Dtype | Destination Dtype | Scale/Offset Dtype | Description |
|----------|-----------|--------------------|------|
| `S8` (`int8_t`) | `FP32` | `FP32` | Converted to FP32 after `UNPK4_B8` unpacking |
| `S16` (`int16_t`) | `FP32` | `FP32` | Converted to FP32 after `UNPK_B16` unpacking |

> `dst`, `scale`, and `offset` must have the same dtype, which must be `FP32`; `src` supports only `S8`/`S16`. Other dtype combinations are invalid (intercepted by `static_assert` in the implementation).

## Implementation Notes

TDEQUANT executes on the vector pipe (`PIPE_V`) and does not require a `tmp` scratch tile (unlike the 5-stage type conversion chain of `TQUANT` on Atlas A2/A3 training products/Atlas A2/A3 inference products):

1. **Load and unpack `src`**: `S8` is unpacked via `UNPK4_B8`, and `S16` via `UNPK_B16`, then converted to FP32 via `vcvt` (on KirinX90, `S8` takes the `US_B8` + interleaved path).
2. **Broadcast load parameters**: `scale` and `offset` are broadcast to the entire row in 32-byte blocks via `vlds ... BRC_B32`.
3. **Compute**: `vsub(dst, src, offset)` followed by `vmul(dst, dst, scale)`, that is, subtract the offset first and then de-scale.

## Encoding

TDEQUANT belongs to the Tile Elementwise Pipeline (TEPL) compound transform class instruction:

```text
BSTART.TEPL TDEQUANT, DataType +
B.DATR(optional) +
B.DIM LB0 +
B.DIM (LB1/LB2 for 2D) +
B.IOT
```

| Field | Value |
|------|------|
| Mode | 3 (compound transform) |
| Function | 11 |
| TileOp | `0x6B` |
| Operands (`B.IOT`) | `dst, src, scale, offset` |

## Constraints

| Constraint | Scope | Reason |
|------|---------|------|
| `dst` and `src` must be row-major | All targets | Dequantization broadcasts parameters per row |
| `dst` and `src` have the same valid shape ($M \times N$) | All targets | Element-wise one-to-one correspondence |
| Valid row count of `scale` and `offset` == valid row count of `dst` | All targets | One set of parameters per row |
| `dst`/`scale`/`offset` dtypes must be identical and be `FP32` | All targets | Dequantization output precision |
| `src` ∈ {`S8`, `S16`} | All targets | Integer quantization code word width |

## Examples

```cpp
// src: int8/int16 quantization code; scale/offset: FP32 parameters per row
TDEQUANT(dstTile, srcTile, scaleTile, offsetTile);
```

For complete ST examples, see `tests/npu/a5/src/st/testcase/tdequant/` (A5), `tests/npu/a2a3/src/st/testcase/tdequant/` (A2/A3), and `tests/cpu/st/testcase/tdequant/` (CPU reference implementation).
