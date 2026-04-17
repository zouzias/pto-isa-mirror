/**
 * TADD vs TADDS Benchmark - 2D tile comparison
 * Shapes: 32x64 (2048 elem) and 1x2048 (same elem count)
 * 
 * UB Address Layout (A2A3 bank conflict free):
 *   TADD:  src0=0x0, src1=0x100 (256B), dst=0x10000 (64KB)
 *   TADDS: src=0x0, dst=0x10000 (64KB)
 */
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace pto;

// ========== TADD 32x64 ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
PTO_INTERNAL void runTAdd(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData src0Global(src0, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData src1Global(src1, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    // A2A3 bank conflict free: src0/src1 256B apart, dst 64KB from sources
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x100);      // 256B offset
    TASSIGN(dstTile, 0x10000);     // 64KB offset

    for (int i = 0; i < 10; i++) {
        TLOAD(src0Tile, src0Global);
        TLOAD(src1Tile, src1Global);
        TADD(dstTile, src0Tile, src1Tile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ========== TADDS 32x64 ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
PTO_INTERNAL void runTAddS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    // src and dst 64KB apart
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);     // 64KB offset

    for (int i = 0; i < 10; i++) {
        TLOAD(srcTile, srcGlobal);
        TADDS(dstTile, srcTile, scalar);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ===== Launch functions =====
extern "C" __global__ AICORE void launchTADD_32x64(__gm__ float *out, __gm__ float *src0, __gm__ float *src1)
{
    runTAdd<float, 32, 64, 32, 64>(out, src0, src1);
}

extern "C" __global__ AICORE void launchTADDS_32x64(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTAddS<float, 32, 64, 32, 64>(out, src, scalar);
}

extern "C" __global__ AICORE void launchTADD_1x2048(__gm__ float *out, __gm__ float *src0, __gm__ float *src1)
{
    runTAdd<float, 1, 2048, 1, 2048>(out, src0, src1);
}

extern "C" __global__ AICORE void launchTADDS_1x2048(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTAddS<float, 1, 2048, 1, 2048>(out, src, scalar);
}

// ===== Template launchers for test framework =====
template <uint32_t caseId>
void launchTADDBenchTestCase(void *out, void *src0, void *src1, aclrtStream stream)
{
    switch (caseId) {
        case 1: launchTADD_32x64<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); break;
        case 2: launchTADD_1x2048<<<1, nullptr, stream>>>((float*)out, (float*)src0, (float*)src1); break;
    }
}

template <uint32_t caseId>
void launchTADDSBenchTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: launchTADDS_32x64<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); break;
        case 2: launchTADDS_1x2048<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar); break;
    }
}

template void launchTADDBenchTestCase<1>(void*, void*, void*, aclrtStream);
template void launchTADDBenchTestCase<2>(void*, void*, void*, aclrtStream);
template void launchTADDSBenchTestCase<1>(void*, void*, float, aclrtStream);
template void launchTADDSBenchTestCase<2>(void*, void*, float, aclrtStream);
