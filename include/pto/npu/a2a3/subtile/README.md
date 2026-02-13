# A2A3 Subtile (1D/2D) Simplified ISA

This folder contains **minimal, subtile** versions of A2/A3 PTO ISA ops with a **new API**.

## Subtile API
- `Subtile1D<T>`: runtime length, **counter mode**, length can be > VL
- `Subtile2D<T>`: runtime rows/cols, **mask + hw repeat**, constraints:
  - `cols <= VL`
  - `rows <= REPEAT_MAX` (<= 255)

## Design goals
- **No row/col loops** in op code
- **Single-instruction path** (counter mode for 1D, repeat+mask for 2D)

## Assumptions
- Row-major only
- For 2D, `rowStride` is in elements
- 2D uses `SetContMaskByDType(cols)` + repeat

Use this for algorithm prototyping or teaching, **not** for general tiling.

## SubtileBrcb (32B block broadcast)
- API: `SubtileBrcb(Subtile2D<T> dst, Subtile1D<T> src)`
- Constraints:
  - `dst.cols == 32B/sizeof(T)` and `dst.rowStride == dst.cols` (contiguous rows)
  - `src.length == dst.rows`
  - `src.length` must be **multiple of 8** (vbrcb repeats)
  - src/dst **32B aligned**

## Demo: emulate TRowExpandSub with SubtileBrcb
Use this when you want block-based bcast (32B per row) with subtile ops:

```cpp
#include <pto/npu/a2a3/subtile/subtile_tile.hpp>
#include <pto/npu/a2a3/subtile/subtile_brcb.hpp>
#include <pto/npu/a2a3/subtile/TSub.hpp>

// src0: [rows, cols], row-major
// src1: [rows] column vector (1 element per row)
// tmp : [rows, elemPerBlock] (32B/row), row-major
// dst : [rows, cols]

constexpr uint32_t elemPerBlock = 32 / sizeof(float);

Subtile2D<float> dst(dstPtr, rows, cols, cols);
Subtile2D<float> src0(src0Ptr, rows, cols, cols);
Subtile2D<float> tmp(tmpPtr, rows, elemPerBlock, elemPerBlock);
Subtile1D<float> src1(src1Ptr, rows); // rows must be multiple of 8

// 1) expand src1 to 32B blocks
SubtileBrcb(tmp, src1);

// 2) row-wise subtract with subtile 2D op (cols <= VL)
TSubSubtile(dst, src0, tmp);
```

Notes:
- If `rows` is not multiple of 8, pad or loop by 8 rows.
- If `cols < elemPerBlock`, `TSubSubtile` uses mask for tail columns.
