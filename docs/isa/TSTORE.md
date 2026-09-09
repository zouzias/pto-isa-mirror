# TSTORE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T05:19:12.809Z pushedAt=2026-08-29T09:05:18.472Z -->

## Instruction Diagram

![TSTORE tile operation](../figures/isa/TSTORE.svg)

## Introduction

Stores data in a tile to a GlobalTensor (GM), optionally using atomic write or quantization parameters.

## Mathematical Semantics

The symbolic representation depends on the shape/stride of the `GlobalTensor` and the layout of the `Tile`. Conceptually (two-dimensional view, with base offset):

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

## Assembly Syntax

Synchronous form:

```text
tstore %t1, %sv_out[%c0, %c0]
```

### AS Level 1 (SSA)

```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2 (DPS)

```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - The source tile position must be one of the following: `TileType::Vec`, `TileType::Mat`, `TileType::Acc`.
    - Runtime: all `dst.GetShape(dim)` values and `src.GetValidRow()/GetValidCol()` must be `> 0`.
    - For source tile position `TileType::Vec` / `TileType::Mat`:
        - `TileData::DType` must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
        - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
        - The layout must match ND/DN/NZ (or special case: `TileData::Rows == 1` or `TileData::Cols == 1`).
        - For `int64_t/uint64_t`, only ND->ND or DN->DN is supported.
    - For the source tile position being `TileType::Acc` (including call forms with quantization parameters and atomic write variants):
        - The destination layout must be ND, NZ, NC1HWC0, or NDC1HWC0.
        - The source data type must be `int32_t` or `float`.
        - When quantization is not used, the destination data type must be `int32_t/float/half/bfloat16_t`.
        - The data type support for ACC-to-GM depends on the call form:

          | Call Form | Source Data Type | Supported Target Data Types |
          | --- | --- | --- |
          | `TSTORE(dst, acc)` | `float` | `float`, `half`, `bfloat16_t` |
          | `TSTORE(dst, acc)` | `int32_t` | `int32_t` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE_FP(dst, acc, fp)` | `float` | `int8_t`, `uint8_t` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE_FP(dst, acc, fp)` | `int32_t` | `int8_t`, `uint8_t`, `half` |

          Other cross-type combinations not listed are not within the supported scope.

        - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND, `1 <= TileData::Rows <= 8192`; if NZ, NC1HWC0, or NDC1HWC0, `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
        - Runtime: `1 <= src.GetValidCol() <= 4095`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - The source tile position must be `TileType::Vec` or `TileType::Acc` (`Mat` storage is not supported on this target).
    - For source tile position `TileType::Vec`:
        - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
        - `TileData::DType` must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`, `float8_e4m3_t`, `float8_e5m2_t`, `hifloat8_t`, `float8_e8m0_t`, `float4_e1m2x2_t`, `float4_e2m1x2_t`.
        - The layout must match ND/DN/NZ (or in special cases: `TileData::Rows == 1` or `TileData::Cols == 1`).
        - Additional alignment constraints are enforced (for example, for ND, the row-major width in bytes must be a multiple of 32; for DN, the column-major height in bytes must be a multiple of 32, with exceptions in special cases).
    - For source tile position `TileType::Acc` (including call forms with quantization parameters and atomic write variants):
        - The target layout must be ND, NZ, NHWC, NCHW, or NCDHW; the source data type must be `int32_t` or `float`.
        - When quantization is not used, the target data type must be `int32_t/float/half/bfloat16_t`.
        - Data type support from ACC to GM depends on the call form:

          | Call Form | Source Data Type | Supported Target Data Types |
          | --- | --- | --- |
          | `TSTORE(dst, acc)` | `float` | `float`, `half`, `bfloat16_t` |
          | `TSTORE(dst, acc)` | `int32_t` | `int32_t` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE_FP(dst, acc, fp)` | `float` | `int8_t`, `uint8_t`, `half`, `bfloat16_t`, `hifloat8_t`, `float8_e4m3_t`, `float` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE_FP(dst, acc, fp)` | `int32_t` | `int8_t`, `uint8_t`, `half`, `bfloat16_t` |

          Other cross-type combinations not listed are not supported.

        - The static shape constraints are the same as the row/column constraints of Atlas A2/A3 training products/Atlas A2/A3 inference products; `AtomicAdd` additionally restricts the target data type to supported atomic types.
- **Valid region**:
    - The implementation uses `src.GetValidRow()`/`src.GetValidCol()` as the transfer size.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TASSIGN(t, 0x1000);
  TSTORE<TileT, GTensor, AtomicType::AtomicAdd>(gout, t);
}
```

## ASM Examples

### Automatic Mode

```text
# Automatic mode: the compiler/runtime is responsible for resource placement and scheduling.
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### Manual Mode

```text
# Manual mode: explicitly bind resources first, then issue the instruction.
# Optional (when the instruction contains tile operands):
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### PTO Assembly Form

```text
tstore %t1, %sv_out[%c0, %c0]
# AS Level 2 (DPS)
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```
