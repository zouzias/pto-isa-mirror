# TCOLARGMAX


## Tile Operation Diagram

![TCOLARGMAX tile operation](../figures/isa/TCOLMAX.svg)

## Introduction

Get the row index of the maximum element for each column.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dst}_{0,j} = \operatorname{argmax}_{0 \le i < R} \mathrm{src}_{i,j} $$

That is, for each column, output the row index where the maximum value resides.

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>
```

Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

Implementation checks (NPU):

- Tile location: `dst` and `src` must be `TileType::Vec`.
- Tile layout: all tiles must be ND fractal (`isRowMajor` and `SLayout::NoneBox`).
- Destination data types: `uint32_t` or `int32_t`.
- `src.ValidCol` must be 1 or -1 (runtime determaxed).
- Runtime valid checks:
    - `src.GetValidCol() == dst.GetValidCol()`.
    - `dst.GetValidRow() == 1`.
    - `src.GetValidRow() != 0` and `src.GetValidCol() != 0`.

- **A2A3**:
  - Source data types: `half`, `float`, `uint16_t`, `uint32_t`.
  - DType consistency: `src.DType == tmp.DType`.
  - `tmp` tile is required and used internally for computation.

- **A5**:
  - Source data types: `half`, `float`, `uint16_t`, `uint32_t`, `int8_t`, `uint8_t`, `int16_t`, `int32_t`.
  - `tmp` tile is not used internally; it is retained only for interface compatibility.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TCOLARGMAX(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TCOLARGMAX(dst, src, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
# pto.tassign %arg2, @tile(0x3000)
%dst = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Instructions

- [TCOLMAX](TCOLMAX.md) - Reduce each column by taking the maximum across rows.
- [TCOLMAX](TCOLMAX.md) - Reduce each column by taking the maximum across rows.
- [TCOLARGMAX](TCOLARGMAX.md) - Get the row index of the maximum element for each column.
- [TROWARGMAX](TROWARGMAX.md) - Get the column index of the maximum element for each row.