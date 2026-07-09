# TADDSC


## Tile Operation Diagram

![TADDSC tile operation](../figures/isa/TADDSC.svg)

## Introduction

Elementwise fused add with scalar and a second tile: `src0 + scalar + src1`.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{scalar} + \mathrm{src1}_{i,j} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDSC(TileData& dst, TileData& src0, typename TileData::DType scalar, TileData& src1,
                            WaitEvents&... events);
```

## Constraints

- **Implementation checks (CPU)**:
    - No explicit data type, location, or layout checks; the implementation uses generic tile iteration.
- **No NPU implementation**: TADDSC is currently supported only on the CPU simulator; there is no A2A3 or A5 backend.
- **Valid region**:
    - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, out;
  TADDSC(out, a, 2.0f, b);
}
```

