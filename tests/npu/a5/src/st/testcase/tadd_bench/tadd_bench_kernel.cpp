/**
 * TADD Benchmark Kernel - Float32 only
 */
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTAddBench(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols),
                         pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));
    GlobalData src0Global(src0, pto::Shape(1, 1, 1, vRows, vCols),
                          pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));
    GlobalData src1Global(src1, pto::Shape(1, 1, 1, vRows, vCols),
                          pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));

    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileData dstTile(vRows, vCols);
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TADD<TileData, TileData, TileData>(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    TADD<TileData, TileData, TileData>(dstTile, dstTile, src1Tile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTAddBench(T *out, T *src0, T *src1, void *stream)
{
    runTAddBench<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

// Float32 instantiations
template void LaunchTAddBench<float, 64, 64, 64, 64>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 8, 512, 8, 512>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 1, 4096, 1, 4096>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 64, 128, 64, 128>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 16, 512, 16, 512>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 1, 8192, 1, 8192>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 128, 128, 128, 128>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 32, 512, 32, 512>(float*, float*, float*, void*);
template void LaunchTAddBench<float, 1, 16384, 1, 16384>(float*, float*, float*, void*);
