# TFMOD


## Tile Operation Diagram

![TFMOD tile operation](../figures/isa/TFMOD.svg)

## Introduction

Elementwise fmod (truncation toward zero) of two tiles.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j})$$

## Assembly Syntax

Synchronous form:

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tfmod %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = FmodAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

`PrecisionType` has the following values available:

* `FmodAlgorithm::DEFAULT`: Normal algorithm, faster but with lower precision.
* `FmodAlgorithm::HIGH_PRECISION`: High precision algorithm, but slower.

## Constraints

- **Implementation checks (A2A3)**:
    - `TileData::DType` must be `float`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
    - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**:
    - `TileData::DType` must be one of: `half`, `float`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
    - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.
- **High Precision Algorithm**:
    - Only available on A5 for `float` type; `PrecisionType` option is ignored on A2A3.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  TFMOD(out, a, b);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tfmod %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfmod %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

