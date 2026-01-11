# TCOLEXPANDDIV

## Introduction

Column-wise broadcast divide: divide each element of `src0` by a per-column vector `src1` broadcast across rows.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{0,j}} $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
%dst = tcolexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst& dst, TileDataDst& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `TileDataDst::DType` must be one of: `half`, `float`.
  - `dst`, `src0`, `src1` must be `TileType::Vec` (compile-time).
  - `dst`, `src0`, `src1` must be `RowMajor` (compile-time).
  - Runtime: `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`.
  - Runtime: `src1.GetValidRow() == 1` and `src1.GetValidCol() == dst.GetValidCol()`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using MatT = Tile<TileType::Vec, float, 16, 64>;
  using RowVecT = Tile<TileType::Vec, float, 1, 64>;
  MatT src0, dst;
  RowVecT src1;
  TCOLEXPANDDIV(dst, src0, src1);
}
```

