# pto.tstore

`pto.tstore` is part of the [Memory And Data Movement](../../memory-and-data-movement.md) instruction set.

## Summary

Store data from a tile into global memory. The transfer is rectangular, spanning `src.GetValidRow()` by `src.GetValidCol()` elements.

## Mechanism

`pto.tstore` initiates a DMA transfer from the source tile buffer to the destination GlobalTensor. The transfer reads a rectangular region from the source tile and writes it to global memory.

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. The transfer size is `R × C` elements. The element mapping depends on the GlobalTensor layout:

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

The operation supports optional atomic accumulation modes and quantization parameters on certain backends.

## Syntax

### PTO Assembly Form

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

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
// Basic store
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events);

// Pre-quantization scalar
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events);

// Fix-pipe quantized store
template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events);
```

The `preQuantScalar` and `TSTORE_FP` overloads are only legal for `TileType::Acc` on A2/A3 and A5 backends. They do not provide a native vec-tile quantized store contract.

## Inputs

| Operand | Description |
|---------|-------------|
| `dst` | Destination GlobalTensor in GM |
| `src` | Source tile. Transfer shape is `src.GetValidRow()` × `src.GetValidCol()`. |
| `atomicType` | Optional atomic mode (e.g., `AtomicAdd`). Default: no atomic behavior. |
| `preQuantScalar` | Optional scalar for pre-quantization (Acc tiles only). |
| `fp` | Optional fix-pipe tile for quantized store (Acc tiles only). |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | `RecordEvent` | Token signaling completion |

After the store completes, the data is written to `dst`. With atomic modes, values are accumulated. With `TSTORE_FP`, the transfer uses the fix-pipe sideband state.

## Side Effects

Writes to global memory. With atomic modes, concurrent access may produce implementation-defined results for individual accumulation operations.

## Constraints

- **Valid region**: Transfer size is `src.GetValidRow()` × `src.GetValidCol()`.
- **Element size match**: `sizeof(tile.dtype) == sizeof(gtensor.dtype)`.
- **Layout compatibility**: Tile layout and GM layout must be a supported combination.
- **Atomic modes**: Only supported on `TileType::Acc`. Supported modes: `AtomicNone`, `AtomicAdd`, `AtomicMax`, `AtomicMin` (A5 only).

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier.
- Programs must not rely on behavior outside the documented legal domain.

## Target-Profile Restrictions

**A2/A3**:
- Source tile location: `TileType::Vec`, `TileType::Mat`, or `TileType::Acc`.
- `Vec`/`Mat`: `sizeof(TileData::DType)` must match `sizeof(GlobalData::DType)`.
- Supported dtypes: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
- `int64_t/uint64_t`: only ND→ND or DN→DN.
- `Acc` (non-quantized): destination dtype must be `__gm__ int32_t/float/half/bfloat16_t`.
- `Acc` shape: `1 <= TileData::Cols <= 4095`; if ND then `1 <= TileData::Rows <= 8192`; if NZ then `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.

**A5**:
- Source tile location: `TileType::Vec` or `TileType::Acc`.
- `Vec`: `sizeof(TileData::DType)` must match `sizeof(GlobalData::DType)`.
- Additional dtypes on A5: `float8_e4m3_t`, `float8_e5m2_t`, `hifloat8_t`, `float4_e1m2x2_t`, `float4_e2m1x2_t`.
- Additional alignment constraints (e.g., ND row-major width in bytes must be a multiple of 32).
- `Acc`: destination layout must be ND or NZ; source dtype must be `int32_t` or `float`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T>
void example(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

## See Also

- Instruction set overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Previous op in instruction set: [pto.tprefetch](./tprefetch.md)
- Next op in instruction set: [pto.tstore_fp](./tstore-fp.md)
