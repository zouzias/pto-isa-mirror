/**
 * Tile Performance Benchmark
 * Supports: TADD, TADDS, TEXP
 * Types: float (fp32), aclFloat16 (fp16)
 * 
 * UB Address Layout (A2A3 bank conflict free):
 *   Binary ops: src0=0x0, src1=0x100 (256B), dst=0x10000 (64KB)
 *   Unary ops:  src=0x0, dst=0x10000 (64KB)
 */
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace pto;

// ========== TADD ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
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
    
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x100);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {
        TLOAD(src0Tile, src0Global);
        TLOAD(src1Tile, src1Global);
        TADD(dstTile, src0Tile, src1Tile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ========== TADDS ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTAddS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {
        TLOAD(srcTile, srcGlobal);
        TADDS(dstTile, srcTile, scalar);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ========== TEXP ==========
template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTExp(__gm__ T *out, __gm__ T *src)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape(vRows, vCols), DynStride(tileH, tileW));
    GlobalData dstGlobal(out, DynShape(vRows, vCols), DynStride(tileH, tileW));
    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {
        TLOAD(srcTile, srcGlobal);
        TEXP(dstTile, srcTile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }
}

// ===== Template launchers with aclFloat16 -> half conversion =====

// TADD launcher
template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADD(void *out, void *src0, void *src1, aclrtStream stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTAdd<half, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
    else
        runTAdd<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((T*)out, (T*)src0, (T*)src1);
}

// TEXP launcher
template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTEXP(void *out, void *src, aclrtStream stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTExp<half, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((half*)out, (half*)src);
    else
        runTExp<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((T*)out, (T*)src);
}

// TADDS launcher - scalar passed as float, kernel casts internally
template <typename T, int tileH, int tileW, int vRows, int vCols>
void launchTADDS(void *out, void *src, float scalar, aclrtStream stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTAddS<half, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((half*)out, (half*)src, (half)scalar);
    else
        runTAddS<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>((T*)out, (T*)src, (T)scalar);
}

// ===== Explicit template instantiations =====

// TADD float - 16KB (4096 elements)
template void launchTADD<float, 1, 4096, 1, 4096>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 64, 64, 64, 64>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 128, 32, 128, 32>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 32, 128, 32, 128>(void*, void*, void*, aclrtStream);
// TADD float - 32KB (8192 elements)
template void launchTADD<float, 1, 8192, 1, 8192>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 64, 128, 64, 128>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 256, 32, 256, 32>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 32, 256, 32, 256>(void*, void*, void*, aclrtStream);
// TADD float - 64KB (16384 elements)
template void launchTADD<float, 1, 16384, 1, 16384>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 128, 128, 128, 128>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 512, 32, 512, 32>(void*, void*, void*, aclrtStream);
template void launchTADD<float, 32, 512, 32, 512>(void*, void*, void*, aclrtStream);

// TADD half - 16KB (8192 elements)
template void launchTADD<aclFloat16, 1, 8192, 1, 8192>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 64, 128, 64, 128>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 256, 32, 256, 32>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 32, 256, 32, 256>(void*, void*, void*, aclrtStream);
// TADD half - 32KB (16384 elements)
template void launchTADD<aclFloat16, 1, 16384, 1, 16384>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 128, 128, 128, 128>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 512, 32, 512, 32>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 32, 512, 32, 512>(void*, void*, void*, aclrtStream);
// TADD half - 64KB (32768 elements)
template void launchTADD<aclFloat16, 1, 32768, 1, 32768>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 128, 256, 128, 256>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 1024, 32, 1024, 32>(void*, void*, void*, aclrtStream);
template void launchTADD<aclFloat16, 32, 1024, 32, 1024>(void*, void*, void*, aclrtStream);

// TEXP float - 16KB (4096 elements)
template void launchTEXP<float, 1, 4096, 1, 4096>(void*, void*, aclrtStream);
template void launchTEXP<float, 64, 64, 64, 64>(void*, void*, aclrtStream);
template void launchTEXP<float, 128, 32, 128, 32>(void*, void*, aclrtStream);
template void launchTEXP<float, 32, 128, 32, 128>(void*, void*, aclrtStream);
// TEXP float - 32KB (8192 elements)
template void launchTEXP<float, 1, 8192, 1, 8192>(void*, void*, aclrtStream);
template void launchTEXP<float, 64, 128, 64, 128>(void*, void*, aclrtStream);
template void launchTEXP<float, 256, 32, 256, 32>(void*, void*, aclrtStream);
template void launchTEXP<float, 32, 256, 32, 256>(void*, void*, aclrtStream);
// TEXP float - 64KB (16384 elements)
template void launchTEXP<float, 1, 16384, 1, 16384>(void*, void*, aclrtStream);
template void launchTEXP<float, 128, 128, 128, 128>(void*, void*, aclrtStream);
template void launchTEXP<float, 512, 32, 512, 32>(void*, void*, aclrtStream);
template void launchTEXP<float, 32, 512, 32, 512>(void*, void*, aclrtStream);

// TEXP half - 16KB (8192 elements)
template void launchTEXP<aclFloat16, 1, 8192, 1, 8192>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 64, 128, 64, 128>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 256, 32, 256, 32>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 32, 256, 32, 256>(void*, void*, aclrtStream);
// TEXP half - 32KB (16384 elements)
template void launchTEXP<aclFloat16, 1, 16384, 1, 16384>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 128, 128, 128, 128>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 512, 32, 512, 32>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 32, 512, 32, 512>(void*, void*, aclrtStream);
// TEXP half - 64KB (32768 elements)
template void launchTEXP<aclFloat16, 1, 32768, 1, 32768>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 128, 256, 128, 256>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 1024, 32, 1024, 32>(void*, void*, aclrtStream);
template void launchTEXP<aclFloat16, 32, 1024, 32, 1024>(void*, void*, aclrtStream);

// TADDS float - 16KB (4096 elements)
template void launchTADDS<float, 1, 4096, 1, 4096>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 64, 64, 64, 64>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 128, 32, 128, 32>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 32, 128, 32, 128>(void*, void*, float, aclrtStream);
// TADDS float - 32KB (8192 elements)
template void launchTADDS<float, 1, 8192, 1, 8192>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 64, 128, 64, 128>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 256, 32, 256, 32>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 32, 256, 32, 256>(void*, void*, float, aclrtStream);
// TADDS float - 64KB (16384 elements)
template void launchTADDS<float, 1, 16384, 1, 16384>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 128, 128, 128, 128>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 512, 32, 512, 32>(void*, void*, float, aclrtStream);
template void launchTADDS<float, 32, 512, 32, 512>(void*, void*, float, aclrtStream);

// TADDS half - 16KB (8192 elements)
template void launchTADDS<aclFloat16, 1, 8192, 1, 8192>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 64, 128, 64, 128>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 256, 32, 256, 32>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 32, 256, 32, 256>(void*, void*, float, aclrtStream);
// TADDS half - 32KB (16384 elements)
template void launchTADDS<aclFloat16, 1, 16384, 1, 16384>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 128, 128, 128, 128>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 512, 32, 512, 32>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 32, 512, 32, 512>(void*, void*, float, aclrtStream);
// TADDS half - 64KB (32768 elements)
template void launchTADDS<aclFloat16, 1, 32768, 1, 32768>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 128, 256, 128, 256>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 1024, 32, 1024, 32>(void*, void*, float, aclrtStream);
template void launchTADDS<aclFloat16, 32, 1024, 32, 1024>(void*, void*, float, aclrtStream);
