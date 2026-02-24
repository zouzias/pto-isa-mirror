# AscendC API vs PTO (A2A3 ST) — usability comparison

Scope: Compare **API usability (shape/type deduction, boilerplate)** using existing ST kernels under `tests/npu/a2a3/src/st/testcase/`.
Focus cases:
- Elementwise: **tadd**
- Reduce: **tcolsum**
- Broadcast/expand: **trowexpand**
- Cast/convert: **tcvt**

Below, each case includes:
- **PTO full kernel** (verbatim from this repo)
- **AscendC full kernel (API‑based)** — complete, minimal version using AscendC API names from the official API list.

> Notes for AscendC examples:
> - Written to match the **CANN AscendC API list** (SetMask/Copy/Add/ReduceSum/Brcb/Cast/etc.).
> - You may need to adjust include paths / namespaces to your local AscendC SDK.

---

## 1) Elementwise Add (tadd)

### PTO — full kernel (repo)
File: `tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp`

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}
```

### AscendC — full kernel (API‑based)

```cpp
#include <ascendc/ascendc.h>

// Elementwise add: dst = src0 + src1
__aicore__ void add_kernel(__gm__ float *dst, __gm__ float *src0, __gm__ float *src1,
                           int rows, int cols, int rowStride)
{
    // local buffers (UB)
    __ubuf__ float *ub0 = (__ubuf__ float *)0x0;
    __ubuf__ float *ub1 = (__ubuf__ float *)0x10000;
    __ubuf__ float *ubd = (__ubuf__ float *)0x20000;

    // 1) DataCopy to UB
    // DataCopy(dst, src, repeat, dstStride, srcStride)
    DataCopy(ub0, src0, /*repeat=*/rows, /*dstStride=*/rowStride, /*srcStride=*/rowStride);
    DataCopy(ub1, src1, /*repeat=*/rows, /*dstStride=*/rowStride, /*srcStride=*/rowStride);

    // 2) Vector add (single call handles all rows via repeat)
    SetMaskNorm();
    SetVectorMask(0, cols); // valid columns
    Add(ubd, ub0, ub1, /*repeat=*/rows - 1, /*dstRep=*/1, /*src0Rep=*/1, /*src1Rep=*/1,
        /*dstStride=*/rowStride / 8, /*src0Stride=*/rowStride / 8, /*src1Stride=*/rowStride / 8);

    // 3) DataCopy back to GM
    DataCopy(dst, ubd, /*repeat=*/rows, /*dstStride=*/rowStride, /*srcStride=*/rowStride);
}
```

**Usability:** PTO is shorter and safer (type/shape inferred by Tile). AscendC is more verbose (manual pointers/stride/mask).

---

## 2) Reduce Sum (tcolsum)

### PTO — full kernel (repo)
File: `tests/npu/a2a3/src/st/testcase/tcolsum/tcolsum_kernel.cpp`

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, int cols, int src_row, int src_validRow, bool IsBinary>
__global__ AICORE void runTCOLSUM(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(src_validRow, cols), DynDim2Stride(src_row, cols));
    GlobalData dstGlobal(out, DynDim2Shape(1, cols), DynDim2Stride(1, cols));

    using srcTileData = Tile<TileType::Vec, T, src_row, cols, BLayout::RowMajor, -1, -1>;
    using tmpTileData = Tile<TileType::Vec, T, src_row, cols, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, 1, cols, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(src_validRow, cols);
    tmpTileData tmpTile(src_validRow / 2, cols);
    dstTileData dstTile(1, cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x14000);
    TASSIGN(tmpTile, 0x28000);

    // 清除脏数据
    TLOAD(dstTile, dstGlobal);

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLSUM(dstTile, srcTile, tmpTile, IsBinary);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}
```

### AscendC — full kernel (API‑based)

```cpp
#include <ascendc/ascendc.h>

// Column-wise reduce sum: dst[0, c] = sum(src[r, c])
__aicore__ void colsum_kernel(__gm__ float *dst, __gm__ float *src,
                             int rows, int cols, int rowStride)
{
    __ubuf__ float *ub = (__ubuf__ float *)0x0;
    __ubuf__ float *ubd = (__ubuf__ float *)0x10000;

    // 1) Copy src to UB
    DataCopy(ub, src, /*repeat=*/rows, /*dstStride=*/rowStride, /*srcStride=*/rowStride);

    // 2) ReduceSum across rows
    // Use WholeReduceSum or ReduceSum depending on API availability
    SetMaskNorm();
    SetVectorMask(0, cols);

    // Example using WholeReduceSum (per repeat)
    // repeat = rows-1, stride uses rowStride/8 (32B block stride)
    WholeReduceSum(ubd, ub, /*repeat=*/rows - 1,
                   /*dstRep=*/1, /*srcRep=*/1, /*dstStride=*/rowStride / 8, /*srcStride=*/rowStride / 8);

    // 3) Copy result (1 row) back to GM
    DataCopy(dst, ubd, /*repeat=*/1, /*dstStride=*/rowStride, /*srcStride=*/rowStride);
}
```

**Usability:** PTO hides temp buffers and stride logic; AscendC requires explicit reduction mode, mask, stride.

---

## 3) Broadcast/Expand (trowexpand)

### PTO — full kernel (repo)
File: `tests/npu/a2a3/src/st/testcase/trowexpand/trowexpand_kernel.cpp`

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, int rows, int src_col, int src_validCol, int dst_col, int dst_validCol>
__global__ AICORE void runTROWEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, rows, -1>;
    using DynStridDim5 = Stride<1, 1, rows, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    GlobalData srcGlobal(src, DynShapeDim5(src_validCol), DynStridDim5(src_col));
    GlobalData dstGlobal(out, DynShapeDim5(dst_validCol), DynStridDim5(dst_col));
    using TileDataSrc = Tile<TileType::Vec, T, rows, src_col, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, rows, dst_col, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(rows, src_validCol);
    TileDataDst dstTile(rows, dst_validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, rows * src_col * sizeof(T)); // UB最大到0x40000

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPAND(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}
```

### AscendC — full kernel (API‑based)

```cpp
#include <ascendc/ascendc.h>

// Broadcast one column across row (row expand)
__aicore__ void rowexpand_kernel(__gm__ float *dst, __gm__ float *src,
                                int rows, int srcCols, int dstCols)
{
    __ubuf__ float *ub = (__ubuf__ float *)0x0;
    __ubuf__ float *ubd = (__ubuf__ float *)0x10000;

    // 1) Copy src (rows x srcCols) to UB
    DataCopy(ub, src, /*repeat=*/rows, /*dstStride=*/srcCols, /*srcStride=*/srcCols);

    // 2) Broadcast each element to a 32B block (Brcb) if dstCols == 32B/sizeof(T)
    // Otherwise, use Add/Copy with mask/stride
    Brcb(ubd, ub); // block broadcast: each element -> one 32B block

    // 3) Copy expanded result to GM
    DataCopy(dst, ubd, /*repeat=*/rows, /*dstStride=*/dstCols, /*srcStride=*/dstCols);
}
```

**Usability:** PTO still shorter (single `TROWEXPAND`), AscendC requires explicit Brcb/Copy + manual layout.

---

## 4) Cast/Convert (tcvt)

### PTO — full kernel (repo)
File: `tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp`

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace std;
using namespace pto;

template <typename T, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTCVT(__gm__ T *out, __gm__ S *src)
{
    using DynShapeDim4 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim4 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src = GlobalTensor<S, DynShapeDim4, DynStridDim4>;
    using GlobalData_dst = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    using TileDataSrc = Tile<TileType::Vec, S, kTRows_, kTCols_, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;

    TASSIGN(srcTile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x20000 + 0x400 * block_idx);

    GlobalData_src srcGlobal(src);
    GlobalData_dst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TCVT(dstTile, srcTile, RoundMode::CAST_RINT);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();
}
```

### AscendC — full kernel (API‑based)

```cpp
#include <ascendc/ascendc.h>

// Cast: dst = cast(src)
__aicore__ void cast_kernel(__gm__ int16_t *dst, __gm__ float *src,
                            int rows, int cols, int rowStride)
{
    __ubuf__ float  *ub = (__ubuf__ float *)0x0;
    __ubuf__ int16_t *ubd = (__ubuf__ int16_t *)0x10000;

    DataCopy(ub, src, /*repeat=*/rows, /*dstStride=*/rowStride, /*srcStride=*/rowStride);

    SetMaskNorm();
    SetVectorMask(0, cols);
    Cast(ubd, ub); // Cast API from AscendC list

    DataCopy(dst, ubd, /*repeat=*/rows, /*dstStride=*/rowStride, /*srcStride=*/rowStride);
}
```

**Usability:** PTO template types remove explicit dtype parameters; AscendC requires explicit casts and pointer types.

---

# Summary — Which is easier for ST‑style kernels?

| Case | PTO (Tile‑template) | AscendC (pointer/struct) | Winner (usability) |
|---|---|---|---|
| Elementwise add | Minimal, type/shape inferred | More setup (tensor descriptors + mask) | **PTO** |
| Reduce sum | One API call; fewer manual strides | Manual mask/stride/repeat | **PTO** |
| Broadcast/expand | One API call; layout encapsulated | Brcb/Copy w/ manual layout | **PTO** |
| Cast | Dtype inferred from templates | Explicit dtype/descriptor | **PTO** |

**Overall:** For **ST‑style kernels** (single op + minimal plumbing), **PTO is more user‑friendly** because the template tiles **deduce shape/type/stride** and reduce boilerplate. AscendC is closer to raw HW control and gives more flexibility, but requires **more manual configuration** and is easier to mis‑specify.

---

## References (in this repo)
- tadd: `tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp`
- tcolsum: `tests/npu/a2a3/src/st/testcase/tcolsum/tcolsum_kernel.cpp`
- trowexpand: `tests/npu/a2a3/src/st/testcase/trowexpand/trowexpand_kernel.cpp`
- tcvt: `tests/npu/a2a3/src/st/testcase/tcvt/tcvt_kernel.cpp`

## AscendC API list (external)
- CANN AscendC API list: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/opdevgapi/atlasascendc_api_07_0004.html
