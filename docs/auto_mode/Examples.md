# Auto Mode Examples

## Scope

This document shows examples of kernels written in auto mode versus in manual mode.


## TADD

```cpp

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

AICORE void runTAdd(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1) {
    using DynShapeDim5 = Shape<1, 1, 1, 64, 64>;
    using DynStridDim5 = Stride<1, 1, 1, 64, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, 64, 64>;
    TileData src0Tile(64, 64);
    TileData src1Tile(64, 64);
    TileData dstTile(64, 64);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    TADD(dstTile, src0Tile, src1Tile);

    TSTORE(dstGlobal, dstTile);
}

```

Whereas in manual mode, it would look like this:

```cpp

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

AICORE void runTAdd(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1) {
    using DynShapeDim5 = Shape<1, 1, 1, 64, 64>;
    using DynStridDim5 = Stride<1, 1, 1, 64, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor, 64, 64>;
    TileData src0Tile(64, 64);
    TileData src1Tile(64, 64);
    TileData dstTile(64, 64);

    /* only in manual mode */
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    /* event model only in manual mode */
    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
}

```