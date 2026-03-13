# TMAXS


## Tile Operation Diagram

![TMAXS tile operation](../figures/isa/TMAXS.svg)

## Introduction

Elementwise max of a tile and a scalar: `max(src, scalar)`.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
%dst = tmaxs %src, %scalar : !pto.tile<...>, f32
```

### IR Level 1 (SSA)

```text
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMAXS(TileData& dst, TileData& src0, typename TileData::DType scalar, WaitEvents&... events);
```

## Constraints

- **Supported Data Types**:
  - A2A3 Architecture: `int32_t`, `int16_t`, `half`, `float`
  - A5 Architecture: `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, `half`, `bfloat16_t`, `uint8_t`, `int8_t`
- **Tile Layout**: Tiles must use row-major layout (`TileData::isRowMajor == true`)
- **Tile Type**: Must be `TileType::Vec`
- **Valid Region**: The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`, with `dst.GetValidRow() > 0` and `dst.GetValidCol() > 0`
- **Input/Output Compatibility**:
  - `dst` and `src` must have the same data type
  - `dst` and `src` must have the same valid dimensions (`dst.GetValidRow() == src.GetValidRow()` and `dst.GetValidCol() == src.GetValidCol()`)
  - Scalar type must match the Tile data type

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TMAXS(out, x, 0.0f);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmaxs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tmaxs %src, %scalar : !pto.tile<...>, f32
# IR Level 2 (DPS)
pto.tmaxs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

