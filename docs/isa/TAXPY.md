# TAXPY

## Tile Operation Diagram

![TAXPY tile operation](../figures/isa/TAXPY.svg)

## Introduction

Scaled elementwise AXPY: scales `src0` by the scalar `scalar` and accumulates it into `dst` (in place). Supports same precision and mixed precision (`dst` of `float`, `src0` of `half`).

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{scalar} \cdot \mathrm{src0}_{i,j} + \mathrm{dst}_{i,j} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TAXPY(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);
```

## Constraints

- The data type of `dst` must be `half` or `float`.
- The data types of `dst` and `src0` must match; or the mixed-precision case — `dst` is `float` and `src0` is `half`.
- The `TileType` of `dst` must be `Vec`.
- `src0` and `dst` must have the same number of valid rows and columns.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto(__gm__ float* in) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  // ... src and dst previously loaded with TLOAD ...
  TAXPY(dst, src, 2.0f);  // dst = 2.0 * src + dst
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  TileT src, dst;
  TASSIGN<0x0000>(src);
  TASSIGN<0x0400>(dst);
  TAXPY(dst, src, static_cast<half>(2.0));
}
```
